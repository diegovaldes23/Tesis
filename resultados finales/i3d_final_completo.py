#!/usr/bin/env python3
"""
i3d_final_completo.py  (autocontenido)
=============================================================================
Corrida FINAL de I3D con TODOS los datos. No depende de ningun otro script.
Entrena las tres configuraciones en la misma carpeta (comparacion justa):

    BASELINE   codificador congelado (sin LoRA) + MLP   -> features cacheadas
    LORA+MLP   codificador con LoRA + MLP entrenables   -> end-to-end
    SOLO_LORA  codificador con LoRA; MLP del baseline congelado -> end-to-end

Incluye ademas, al final, las METRICAS OPERACIONALES (GFLOPs, latencia,
throughput y overhead de LoRA), con el mismo criterio que TimeSformer/X-CLIP.

CONFIG DE LoRA: r=16, alpha=16 (alpha=r), dropout=0.24
  (la que GANO en la calibracion con Optuna; ver anexo de calibracion).

LECTURA RAPIDA:
  - Lee los videos desde disco LOCAL (/tmp) en vez del NFS (remapeo de path).
  - Si existen los .npy pre-extraidos del subset, los usa (mas rapido aun);
    si no, decodifica el video local. Cae al NFS solo si falta el local.
  - BASELINE cachea embeddings (encoder congelado) -> entrena el MLP en segundos.
  - LoRA+MLP y Solo LoRA son end-to-end (el encoder se entrena): NO cacheables.

Ejecutar:
    python3 -u i3d_final_completo.py 2>&1 | tee i3d_final.log
=============================================================================
"""

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message=r".*torch\.cuda\.amp.*")
warnings.filterwarnings("ignore", message=r".*GradScaler.*")
warnings.filterwarnings("ignore", message=r".*autocast.*")

import sys
import json
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader, TensorDataset
from sklearn.metrics import (roc_auc_score, accuracy_score, precision_score,
                             recall_score, f1_score)


# ═══════════════════════════════════════════════════════════════════════════
# PARAMETROS
# ═══════════════════════════════════════════════════════════════════════════
# Config de LoRA GANADORA (Optuna). alpha=16 con r=16 => alpha=r.
LORA_RANK, LORA_ALPHA, LORA_DROPOUT = 16, 16, 0.24

# Que configuraciones correr (para repartir en 2 GPUs si quieres).
HACER_BASELINE  = True
HACER_LORA_MLP  = True
HACER_SOLO_LORA = True

# Velocidad
BATCH_VID    = 24          # RTX6000=24GB; baja a 16 si hay OOM
NUM_WORKERS  = 28          # 40 nucleos; mas workers ayudan a la lectura
EPOCHS       = 50
PATIENCE     = 8
LOG_EVERY    = 50          # imprime "ep X train i/N" cada N batches

CHEQUEAR_LORA = True       # confirma que LoRA aprende antes de gastar computo
HACER_OPERACIONAL = True   # mide GFLOPs/latencia/overhead al final

# Lectura desde disco local
PATH_NFS      = "/home/DIINF/dvaldes/tesis/UCF_Crime"
PATH_LOCAL    = "/tmp/dvaldes/UCF_Crime"        # None -> leer del NFS
CLIPS_NPY_DIR = Path("/tmp/dvaldes/clips_npy")  # None -> no usar .npy

OUT_DIR = "processed/i3d_final_r16a16d024"

# Entrenamiento del MLP sobre features cacheadas (baseline). Barato.
MLP_CACHE_EPOCHS, MLP_CACHE_PATIENCE, MLP_CACHE_BATCH = 100, 12, 512


# ═══════════════════════════════════════════════════════════════════════════
# CONFIG BASE (copiada de train_i3d_v2_sincalib.py)
# ═══════════════════════════════════════════════════════════════════════════
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True

CFG = {
    "index_path":   "processed/index_clips.csv",
    "i3d_repo":     "/home/DIINF/dvaldes/pytorch-i3d",
    "weights_path": "/home/DIINF/dvaldes/models/i3d/rgb_imagenet.pt",
    "num_frames": 32, "img_size": 224,
    "mean": [0.43216, 0.394666, 0.37645],
    "std":  [0.22803, 0.22145, 0.216989],
    "mlp_hidden": 128, "mlp_dropout": 0.3,
    "use_amp": True, "lr_mlp": 1e-3, "lr_e2e": 1e-4,
    "lora_targets": ["Mixed_5b", "Mixed_5c"],
    "seed": 42,
}
EMB_DIM = 1024
Path(OUT_DIR).mkdir(parents=True, exist_ok=True)


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ─── I3D ────────────────────────────────────────────────────────────────────
sys.path.append(CFG["i3d_repo"])
from pytorch_i3d import InceptionI3d


def load_i3d(freeze=True):
    model = InceptionI3d(400, in_channels=3)
    model.load_state_dict(torch.load(CFG["weights_path"], map_location=DEVICE))
    if freeze:
        for p in model.parameters():
            p.requires_grad = False
    model.eval()
    return model.to(DEVICE)


# ─── LoRA ───────────────────────────────────────────────────────────────────
class LoRAConv3d1x1(nn.Module):
    def __init__(self, conv, rank, alpha, dropout):
        super().__init__()
        device = conv.weight.device
        self.conv = conv
        for p in self.conv.parameters():
            p.requires_grad = False
        self.scale  = alpha / rank
        self.lora_A = nn.Parameter(torch.randn(rank, conv.in_channels, device=device) * 0.01)
        self.lora_B = nn.Parameter(torch.zeros(conv.out_channels, rank, device=device))
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        base = self.conv(x)
        B, C, Tf, H, W = x.shape
        xr   = self.dropout(x).permute(0, 2, 3, 4, 1).reshape(-1, C)
        lora = xr @ self.lora_A.T @ self.lora_B.T
        lora = lora.reshape(B, Tf, H, W, -1).permute(0, 4, 1, 2, 3)
        return base + lora * self.scale


def aplicar_lora(model, rank, alpha, dropout, targets):
    replaced = 0
    for tname in targets:
        module = getattr(model, tname, None)
        if module is None:
            print(f"  [WARN] bloque '{tname}' no encontrado")
            continue
        for name, child in list(module.named_modules()):
            if isinstance(child, nn.Conv3d) and child.kernel_size == (1, 1, 1):
                parent = module
                parts  = name.split(".")
                for part in parts[:-1]:
                    parent = getattr(parent, part)
                setattr(parent, parts[-1], LoRAConv3d1x1(child, rank, alpha, dropout))
                replaced += 1
    return replaced


def embed(model, x):
    feats = []
    h = model.avg_pool.register_forward_hook(lambda m, i, o: feats.append(o))
    _ = model(x)
    h.remove()
    return feats[0].mean(dim=[2, 3, 4])


class AnomalyMLP(nn.Module):
    def __init__(self, d=EMB_DIM, h=CFG["mlp_hidden"], drop=CFG["mlp_dropout"]):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d, h), nn.BatchNorm1d(h),
            nn.ReLU(inplace=True), nn.Dropout(drop), nn.Linear(h, 1))

    def forward(self, x): return self.net(x)


class FullPipeline(nn.Module):
    def __init__(self, encoder, mlp):
        super().__init__()
        self.encoder = encoder
        self.mlp = mlp

    def forward(self, x): return self.mlp(embed(self.encoder, x))


def modo_train(full):
    full.encoder.eval()
    for m in full.encoder.modules():
        if isinstance(m, LoRAConv3d1x1):
            m.train()
    full.mlp.train()


# ─── Datos (con lectura desde local + .npy) ─────────────────────────────────
def remap_path(p):
    if PATH_LOCAL is None:
        return p
    if str(p).startswith(PATH_NFS):
        local = str(p).replace(PATH_NFS, PATH_LOCAL, 1)
        if Path(local).exists():
            return local
    return p


def sample_frames(video_path, start_frame, end_frame, num_frames):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise IOError(f"No se pudo abrir: {video_path}")
    indices = np.linspace(start_frame, end_frame - 1, num_frames, dtype=int)
    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ret, frame = cap.read()
        if not ret:
            frame = frames[-1] if frames else np.zeros((256, 256, 3), np.uint8)
        else:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)
    cap.release()
    return np.stack(frames)


class VideoClipDataset(Dataset):
    def __init__(self, df):
        self.df = df.reset_index(drop=True)
        self.num_frames = CFG["num_frames"]
        self.img_size = CFG["img_size"]
        self.use_npy = (CLIPS_NPY_DIR is not None and "gidx" in self.df.columns)
        self.transform = T.Compose([
            T.ToTensor(), T.Normalize(mean=CFG["mean"], std=CFG["std"])])

    def __len__(self): return len(self.df)

    def _preprocess(self, frames):
        out = []
        for f in frames:
            if f.shape[:2] != (self.img_size, self.img_size):
                f = cv2.resize(f, (self.img_size, self.img_size))
            out.append(self.transform(f))
        return torch.stack(out, dim=1)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        try:
            frames = None
            if self.use_npy:
                npy = CLIPS_NPY_DIR / f"clip_{int(row['gidx'])}.npy"
                if npy.exists():
                    frames = np.load(npy)
            if frames is None:
                frames = sample_frames(remap_path(row["path"]),
                                       int(row["start_frame"]),
                                       int(row["end_frame"]), self.num_frames)
            clip = self._preprocess(frames)
        except Exception:
            clip = torch.zeros(3, self.num_frames, self.img_size, self.img_size)
        return clip, float(row["y"])


def load_splits():
    df = pd.read_csv(CFG["index_path"]).reset_index().rename(columns={"index": "gidx"})
    print(f"\n  Total clips: {len(df)} | {dict(df['split'].value_counts())}")
    return (df[df["split"] == "train"].reset_index(drop=True),
            df[df["split"] == "val"].reset_index(drop=True),
            df[df["split"] == "test"].reset_index(drop=True))


def make_loaders(df_tr, df_va, df_te):
    kw = dict(num_workers=NUM_WORKERS, pin_memory=(DEVICE.type == "cuda"))
    if NUM_WORKERS > 0:
        kw["persistent_workers"] = True
        kw["prefetch_factor"] = 6
    return (
        DataLoader(VideoClipDataset(df_tr), BATCH_VID, shuffle=True,  **kw),
        DataLoader(VideoClipDataset(df_va), BATCH_VID, shuffle=False, **kw),
        DataLoader(VideoClipDataset(df_te), BATCH_VID, shuffle=False, **kw),
    )


def compute_metrics(probs, labels, thr=0.5):
    preds = (probs >= thr).astype(int)
    fp = int(((preds == 1) & (labels == 0)).sum())
    tn = int(((preds == 0) & (labels == 0)).sum())
    far = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    return {
        "AUC": float(roc_auc_score(labels, probs)),
        "Accuracy": float(accuracy_score(labels, preds)),
        "Precision": float(precision_score(labels, preds, zero_division=0)),
        "Recall": float(recall_score(labels, preds, zero_division=0)),
        "F1": float(f1_score(labels, preds, zero_division=0)),
        "FAR": float(far),
    }


# ─── Entrenamiento end-to-end (con avance sin barras) ───────────────────────
def train_endtoend(full, tr_ldr, va_ldr, lr, epochs, patience, ckpt_path, label):
    crit = nn.BCEWithLogitsLoss()
    opt  = torch.optim.Adam([p for p in full.parameters() if p.requires_grad], lr=lr)
    use_amp = CFG["use_amp"] and DEVICE.type == "cuda"
    scaler  = torch.cuda.amp.GradScaler(enabled=use_amp)
    best_auc, no_imp = 0.0, 0
    n_train = sum(p.numel() for p in full.parameters() if p.requires_grad)
    print(f"\n  [{label}] entrenables: {n_train:,} | lr={lr} | "
          f"epochs={epochs} | patience={patience} | amp={use_amp}")
    print(f"  {'Ep':>3} | {'TrainLoss':>10} | {'ValAUC':>8} | {'No+':>4} | {'Time':>7}")
    print("  " + "-" * 48)

    t0_total = time.time()
    for ep in range(1, epochs + 1):
        t0 = time.time()
        modo_train(full)
        tr_loss = 0.0
        n_tr = len(tr_ldr)
        for i, (xb, yb) in enumerate(tr_ldr, 1):
            xb, yb = xb.to(DEVICE, non_blocking=True), yb.to(DEVICE, non_blocking=True)
            opt.zero_grad()
            with torch.cuda.amp.autocast(enabled=use_amp):
                loss = crit(full(xb).squeeze(1), yb)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            tr_loss += loss.item() * len(yb)
            if i % LOG_EVERY == 0 or i == n_tr:
                print(f"    ep {ep} train {i}/{n_tr} | loss {loss.item():.4f}", flush=True)
        tr_loss /= len(tr_ldr.dataset)

        full.eval()
        all_p, all_l = [], []
        with torch.no_grad():
            for xb, yb in va_ldr:
                with torch.cuda.amp.autocast(enabled=use_amp):
                    out = full(xb.to(DEVICE, non_blocking=True)).squeeze(1)
                all_p.append(torch.sigmoid(out.float()).cpu().numpy())
                all_l.append(yb.numpy())
        va_auc = roc_auc_score(np.concatenate(all_l), np.concatenate(all_p))

        flag = ""
        if va_auc > best_auc:
            best_auc, no_imp = va_auc, 0
            torch.save(full.state_dict(), ckpt_path)
            flag = "best"
        else:
            no_imp += 1
        print(f"  {ep:>3} | {tr_loss:>10.4f} | {va_auc:>8.4f} | "
              f"{no_imp:>4} | {time.time()-t0:>6.1f}s {flag}")
        if no_imp >= patience:
            print(f"  Early stopping en epoca {ep}")
            break

    print(f"  Mejor AUC val: {best_auc:.4f} | "
          f"tiempo: {(time.time()-t0_total)/60:.1f} min")
    return best_auc


@torch.no_grad()
def eval_test(full, te_ldr):
    full.eval()
    use_amp = CFG["use_amp"] and DEVICE.type == "cuda"
    all_p, all_l = [], []
    for xb, yb in te_ldr:
        with torch.cuda.amp.autocast(enabled=use_amp):
            out = full(xb.to(DEVICE, non_blocking=True)).squeeze(1)
        all_p.append(torch.sigmoid(out.float()).cpu().numpy())
        all_l.append(yb.numpy())
    return compute_metrics(np.concatenate(all_p), np.concatenate(all_l))


# ─── Baseline cacheado (features fijas, encoder congelado) ──────────────────
@torch.no_grad()
def extraer_features(enc, ldr, desc):
    enc.eval()
    use_amp = CFG["use_amp"] and DEVICE.type == "cuda"
    feats, labels = [], []
    n = len(ldr)
    for i, (xb, yb) in enumerate(ldr, 1):
        with torch.cuda.amp.autocast(enabled=use_amp):
            e = embed(enc, xb.to(DEVICE, non_blocking=True))
        feats.append(e.float().cpu())
        labels.append(yb)
        if i % LOG_EVERY == 0 or i == n:
            print(f"    {desc} {i}/{n}", flush=True)
    return torch.cat(feats), torch.cat(labels)


def entrenar_mlp_cache(Xtr, ytr, Xva, yva, ckpt_path):
    mlp = AnomalyMLP().to(DEVICE)
    opt = torch.optim.Adam(mlp.parameters(), lr=CFG["lr_mlp"])
    crit = nn.BCEWithLogitsLoss()
    ld = DataLoader(TensorDataset(Xtr, ytr), batch_size=MLP_CACHE_BATCH, shuffle=True)
    Xva_d = Xva.to(DEVICE)
    best_auc, no_imp = 0.0, 0
    print(f"  [BASELINE/MLP-cache] epochs={MLP_CACHE_EPOCHS} patience={MLP_CACHE_PATIENCE}")
    for ep in range(1, MLP_CACHE_EPOCHS + 1):
        mlp.train()
        for xb, yb in ld:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad()
            crit(mlp(xb).squeeze(1), yb).backward()
            opt.step()
        mlp.eval()
        with torch.no_grad():
            p = torch.sigmoid(mlp(Xva_d).squeeze(1).float()).cpu().numpy()
        auc = roc_auc_score(yva.numpy(), p)
        if auc > best_auc:
            best_auc, no_imp = auc, 0
            torch.save({f"mlp.{k}": v for k, v in mlp.state_dict().items()}, ckpt_path)
        else:
            no_imp += 1
        if no_imp >= MLP_CACHE_PATIENCE:
            break
    print(f"  Mejor AUC val (baseline): {best_auc:.4f}")
    return best_auc


@torch.no_grad()
def eval_mlp_cache(ckpt_path, Xte, yte):
    mlp = AnomalyMLP().to(DEVICE)
    state = torch.load(ckpt_path, map_location=DEVICE)
    mlp.load_state_dict({k.replace("mlp.", "", 1): v for k, v in state.items()})
    mlp.eval()
    p = torch.sigmoid(mlp(Xte.to(DEVICE)).squeeze(1).float()).cpu().numpy()
    return compute_metrics(p, yte.numpy())


# ─── Chequeo: ¿LoRA aprende? ────────────────────────────────────────────────
def chequeo_lora():
    print("\n" + "=" * 64)
    print("  CHEQUEO — ¿LoRA aprende en I3D?")
    print("=" * 64)
    enc = load_i3d(freeze=True)
    replaced = aplicar_lora(enc, LORA_RANK, LORA_ALPHA, LORA_DROPOUT, CFG["lora_targets"])
    print(f"  Capas LoRA aplicadas: {replaced}")
    if replaced == 0:
        print("  [ABORTA] No se aplico LoRA."); sys.exit(1)
    lora_B = {n: p for n, p in enc.named_parameters()
              if n.endswith("lora_B") and p.requires_grad}
    antes = max(p.detach().norm().item() for p in lora_B.values())
    mlp = AnomalyMLP().to(DEVICE)
    full = FullPipeline(enc, mlp).to(DEVICE)
    opt = torch.optim.Adam([p for p in full.parameters() if p.requires_grad], lr=1e-3)
    crit = nn.BCEWithLogitsLoss()
    modo_train(full)
    for _ in range(8):
        x = torch.randn(2, 3, CFG["num_frames"], CFG["img_size"], CFG["img_size"], device=DEVICE)
        y = torch.randint(0, 2, (2,), device=DEVICE).float()
        opt.zero_grad()
        crit(full(x).squeeze(1), y).backward()
        opt.step()
    despues = max(p.detach().norm().item() for p in lora_B.values())
    del enc, mlp, full
    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()
    if despues - antes > 1e-8:
        print(f"  OK: lora_B se movio (0 -> {despues:.3e}). LoRA aprende.\n")
    else:
        print("\n  [ABORTA] lora_B no cambio; LoRA no recibe gradiente."); sys.exit(1)


# ─── Metricas operacionales ─────────────────────────────────────────────────
def medir_operacional():
    print("\n" + "=" * 64)
    print("  METRICAS OPERACIONALES (GFLOPs, latencia, overhead LoRA)")
    print("=" * 64)
    N_WARMUP, N_REPS, STRIDE = 10, 100, 16
    UMBRAL = 30 / STRIDE   # 1.875 clips/s == 533 ms/clip

    def calcular_gflops(enc, mlp):
        from fvcore.nn import FlopCountAnalysis
        dummy = torch.randn(1, 3, CFG["num_frames"], CFG["img_size"], CFG["img_size"]).to(DEVICE)
        try:
            f = FlopCountAnalysis(FullPipeline(enc, mlp).eval(), dummy)
        except Exception:
            f = FlopCountAnalysis(enc, dummy)
        f.unsupported_ops_warnings(False)
        f.uncalled_modules_warnings(False)
        return round(f.total() / 1e9, 2)

    @torch.no_grad()
    def medir_latencia(enc, mlp):
        enc.eval(); mlp.eval()
        dummy = torch.randn(1, 3, CFG["num_frames"], CFG["img_size"], CFG["img_size"]).to(DEVICE)
        for _ in range(N_WARMUP):
            mlp(embed(enc, dummy))
        if DEVICE.type == "cuda":
            torch.cuda.synchronize()
        times = []
        for _ in range(N_REPS):
            t0 = time.perf_counter()
            mlp(embed(enc, dummy))
            if DEVICE.type == "cuda":
                torch.cuda.synchronize()
            times.append(time.perf_counter() - t0)
        lat = float(np.mean(times) * 1000)
        return round(lat, 2), round(1000.0 / lat, 2)

    def perfil(con_lora):
        enc = load_i3d(freeze=True)
        n_lora = 0
        if con_lora:
            aplicar_lora(enc, LORA_RANK, LORA_ALPHA, LORA_DROPOUT, CFG["lora_targets"])
            n_lora = sum(p.numel() for p in enc.parameters() if p.requires_grad)
        mlp = AnomalyMLP().to(DEVICE)
        g = calcular_gflops(enc, mlp)
        lat, cps = medir_latencia(enc, mlp)
        del enc, mlp
        if DEVICE.type == "cuda":
            torch.cuda.empty_cache()
        return {"gflops": g, "lat_ms": lat, "clips_s": cps, "lora_params": n_lora}

    print("  Sin LoRA (Baseline)...")
    b = perfil(False)
    print(f"    GFLOPs={b['gflops']} ms/clip={b['lat_ms']} clips/s={b['clips_s']}")
    print("  Con LoRA...")
    l = perfil(True)
    print(f"    GFLOPs={l['gflops']} ms/clip={l['lat_ms']} clips/s={l['clips_s']} "
          f"| LoRA params={l['lora_params']:,}")

    d = round(l["gflops"] - b["gflops"], 3)
    pct = round(100 * d / b["gflops"], 3) if b["gflops"] else 0.0
    print(f"\n  Overhead LoRA: +{d} GFLOPs ({pct}%) | +{l['lora_params']:,} parametros")
    print(f"\n  {'Config':<12}{'GFLOPs':>9}{'ms/clip':>10}{'clips/s':>10}{'TR':>6}")
    print("  " + "-" * 47)
    for name, r in [("Baseline", b), ("LoRA+MLP", l), ("Solo LoRA", l)]:
        tr = "si" if r["clips_s"] >= UMBRAL else "NO"
        print(f"  {name:<12}{r['gflops']:>9.2f}{r['lat_ms']:>10.2f}{r['clips_s']:>10.2f}{tr:>6}")
    print(f"\n  Umbral tiempo real: {UMBRAL:.3f} clips/s (== 533 ms/clip)")

    return {"baseline": b, "con_lora": l,
            "overhead_lora": {"gflops": d, "pct": pct, "params": l["lora_params"]}}


# ═══════════════════════════════════════════════════════════════════════════
def main():
    set_seed(CFG["seed"])
    t0 = time.time()
    out_dir = Path(OUT_DIR)

    print("=" * 64)
    print("  I3D FINAL COMPLETO (baseline + LoRA+MLP + Solo LoRA)")
    print(f"  Device: {DEVICE} | batch={BATCH_VID} workers={NUM_WORKERS} "
          f"| epochs={EPOCHS} patience={PATIENCE}")
    print(f"  LoRA: r={LORA_RANK} alpha={LORA_ALPHA} dropout={LORA_DROPOUT}")
    print(f"  Lectura local: {PATH_LOCAL} | npy: {CLIPS_NPY_DIR}")
    print(f"  Salida: {out_dir}")
    print("=" * 64)

    if CHEQUEAR_LORA:
        chequeo_lora()

    df_tr, df_va, df_te = load_splits()
    tr_ldr, va_ldr, te_ldr = make_loaders(df_tr, df_va, df_te)
    ckpt_base = out_dir / "baseline.pth"
    out = {}

    # BASELINE (cacheado)
    if HACER_BASELINE:
        print("\n  >>> BASELINE (features cacheadas)")
        enc = load_i3d(freeze=True)
        Xtr, ytr = extraer_features(enc, tr_ldr, "feat train")
        Xva, yva = extraer_features(enc, va_ldr, "feat val")
        Xte, yte = extraer_features(enc, te_ldr, "feat test")
        entrenar_mlp_cache(Xtr, ytr, Xva, yva, ckpt_base)
        out["BASELINE"] = eval_mlp_cache(ckpt_base, Xte, yte)
        print(f"  Test: {out['BASELINE']}")
        del enc, Xtr, ytr, Xva, yva, Xte, yte
        if DEVICE.type == "cuda":
            torch.cuda.empty_cache()

    # LORA+MLP
    if HACER_LORA_MLP:
        print("\n  >>> LORA+MLP")
        enc = load_i3d(freeze=True)
        aplicar_lora(enc, LORA_RANK, LORA_ALPHA, LORA_DROPOUT, CFG["lora_targets"])
        mlp = AnomalyMLP().to(DEVICE)
        full = FullPipeline(enc, mlp).to(DEVICE)
        ckpt = out_dir / "lora_mlp.pth"
        train_endtoend(full, tr_ldr, va_ldr, lr=CFG["lr_e2e"],
                       epochs=EPOCHS, patience=PATIENCE, ckpt_path=ckpt, label="LORA+MLP")
        full.load_state_dict(torch.load(ckpt, map_location=DEVICE))
        out["LORA+MLP"] = eval_test(full, te_ldr)
        print(f"  Test: {out['LORA+MLP']}")
        del enc, mlp, full
        if DEVICE.type == "cuda":
            torch.cuda.empty_cache()

    # SOLO LORA (MLP del baseline, congelado)
    if HACER_SOLO_LORA:
        print("\n  >>> SOLO_LORA")
        if not ckpt_base.exists():
            raise FileNotFoundError(
                f"Solo LoRA necesita el MLP del baseline ({ckpt_base}). "
                "Corre el baseline primero (HACER_BASELINE=True).")
        enc = load_i3d(freeze=True)
        aplicar_lora(enc, LORA_RANK, LORA_ALPHA, LORA_DROPOUT, CFG["lora_targets"])
        mlp = AnomalyMLP().to(DEVICE)
        st = torch.load(ckpt_base, map_location=DEVICE)
        mlp.load_state_dict({k.replace("mlp.", "", 1): v for k, v in st.items()})
        for p in mlp.parameters():
            p.requires_grad = False
        full = FullPipeline(enc, mlp).to(DEVICE)
        ckpt = out_dir / "solo_lora.pth"
        train_endtoend(full, tr_ldr, va_ldr, lr=CFG["lr_e2e"],
                       epochs=EPOCHS, patience=PATIENCE, ckpt_path=ckpt, label="SOLO_LORA")
        full.load_state_dict(torch.load(ckpt, map_location=DEVICE))
        out["SOLO_LORA"] = eval_test(full, te_ldr)
        print(f"  Test: {out['SOLO_LORA']}")
        del enc, mlp, full
        if DEVICE.type == "cuda":
            torch.cuda.empty_cache()

    operacional = medir_operacional() if HACER_OPERACIONAL else None

    payload = {
        "config_lora": {"rank": LORA_RANK, "alpha": LORA_ALPHA, "dropout": LORA_DROPOUT},
        "mlp": {"hidden": CFG["mlp_hidden"], "dropout": CFG["mlp_dropout"]},
        "epochs": EPOCHS, "patience": PATIENCE, "batch_vid": BATCH_VID,
        "test": out, "operacional": operacional,
    }
    (out_dir / "i3d_final_r16a16d024_resultados.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False))

    print("\n" + "=" * 64)
    print("  RESUMEN FINAL (test)")
    print("=" * 64)
    print(f"  Config LoRA: r={LORA_RANK} alpha={LORA_ALPHA} dropout={LORA_DROPOUT}")
    for nombre, m in out.items():
        print(f"  {nombre:<10} AUC={m['AUC']:.4f} F1={m['F1']:.4f} FAR={m['FAR']:.4f}")
    print(f"\n  Guardado en: {out_dir / 'i3d_final_r16a16d024_resultados.json'}")
    print(f"  Tiempo total: {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
