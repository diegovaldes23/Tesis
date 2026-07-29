#!/usr/bin/env python3
"""
=============================================================================
  TimeSformer — Detección de Anomalías en Video
  Dataset  : UCF-Crime (processed/index_clips.csv)
  Pipeline : Video → TimeSformer → CLS embedding → MLP → pred binaria

  3 experimentos (igual que I3D):
    1. BASELINE  : TimeSformer congelado + MLP (sobre embeddings)
    2. LORA+MLP  : TimeSformer+LoRA end-to-end + MLP (lr reducido 1e-4)
    3. SOLO_LORA : MLP congelado del baseline + solo LoRA aprende

  Configuración MLP (calibrada sobre I3D):
    arch=fc | hidden_dim=128 | dropout=0.3 | lr=1e-3

  Configuración LoRA (calibrada sobre I3D):
    rank=16 | alpha=16 | lora_dropout=0.24
    Capas objetivo: qkv (proyección Q+K+V combinada en TimeSformer HF)

  CHECKPOINTS DE FASE — retoma desde donde quedó si falla:
    rm processed/ts_results/.phase_emb_done        → re-extraer embeddings
    rm processed/ts_results/.phase_baseline_done   → re-entrenar baseline
    rm processed/ts_results/.phase_lora_done       → re-entrenar LoRA+MLP
    rm processed/ts_results/.phase_solo_lora_done  → re-entrenar Solo LoRA
    rm processed/ts_results/.phase_*               → todo desde cero

  Salidas en processed/ts_results/:
    ts_baseline_best.pth    — checkpoint baseline
    ts_lora_best.pth        — checkpoint LoRA+MLP
    ts_solo_lora_best.pth   — checkpoint Solo LoRA
    ts_results.json         — métricas completas
    ts_figures/             — gráficos para tesis
=============================================================================
Ejecutar:
    screen -S ts
    python3 timesformer_v3.py 2>&1 | tee ts.log
    Ctrl+A D  (para desconectarse sin matar)
    tail -f ts.log  (para seguirlo desde otra terminal)
=============================================================================
"""

# ─────────────────────────────────────────────────────────────────────────────
# SECCIÓN 0 — Imports
# ─────────────────────────────────────────────────────────────────────────────
import json
import time
import warnings
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score, confusion_matrix, f1_score,
    precision_score, recall_score, roc_auc_score, roc_curve,
)
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
from transformers import TimesformerModel

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# SECCIÓN 1 — Configuración global
# ─────────────────────────────────────────────────────────────────────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")

INDEX_PATH       = Path("processed/index_clips.csv")
TIMESFORMER_CKPT = "facebook/timesformer-base-finetuned-k400"
OUT_DIR          = Path("processed/timesformer_final_r16a16d024")
FIG_DIR          = OUT_DIR / "ts_figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Lectura desde disco LOCAL (mucho mas rapido que el NFS) ───────────────────
PATH_NFS   = "/home/DIINF/dvaldes/tesis/UCF_Crime"
PATH_LOCAL = "/dev/shm/dvaldes/UCF_Crime"      # pon None para leer del NFS

# ── Lectura desde disco LOCAL (mucho mas rapido que el NFS) ───────────────────
PATH_NFS    = "/home/DIINF/dvaldes/tesis/UCF_Crime"
PATH_LOCAL  = "/dev/shm/dvaldes/UCF_Crime"      # pon None para leer del NFS
FIG_DIR.mkdir(parents=True, exist_ok=True)

# ── Configuración MLP calibrada sobre I3D ────────────────────────────────────
MLP_CFG = {
    "arch":       "fc",
    "hidden_dim": 128,
    "dropout":    0.3,
    "lr":         1e-3,    # lr para MLP sobre embeddings
    "lr_e2e":     1e-4,    # lr reducido para entrenamiento end-to-end con LoRA
    "epochs":     50,
    "patience":   8,
    "batch_size": 64,
    "batch_vid":  8,       # batch para video crudo (limitado por VRAM)
    "seed":       42,
}

# ── Configuración LoRA (misma que I3D: calibrada con Optuna, r=16) ────────────
# NOTA: en TimeSformer HuggingFace las proyecciones Q,K,V están
# fusionadas en una sola capa llamada "qkv" (768 → 2304)
LORA_CFG = {
    "rank":         16,
    "alpha":        16,
    "lora_dropout": 0.24,
    "target_mods":  ("qkv",),   # nombre real en HF TimeSformer
}

# ── Parámetros de video ───────────────────────────────────────────────────────
T           = 8
IMG_SIZE    = 224
NUM_WORKERS = 16
IMG_MEAN    = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMG_STD     = np.array([0.229, 0.224, 0.225], dtype=np.float32)
EMB_DIM     = 768   # TimeSformer ViT-Base CLS token


def set_seed(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


set_seed(MLP_CFG["seed"])

# ─────────────────────────────────────────────────────────────────────────────
# SECCIÓN 2 — Dataset
# ─────────────────────────────────────────────────────────────────────────────
assert INDEX_PATH.exists(), f"No se encuentra: {INDEX_PATH}"
df_clips = pd.read_csv(INDEX_PATH)
print(f"Clips totales: {len(df_clips):,}")
print(df_clips["split"].value_counts().to_string())

df_train = df_clips[df_clips["split"] == "train"].reset_index(drop=True)
df_val   = df_clips[df_clips["split"] == "val"].reset_index(drop=True)
df_test  = df_clips[df_clips["split"] == "test"].reset_index(drop=True)


def uniform_sample_indices(start_f, end_f, T):
    n   = max(1, end_f - start_f)
    idx = np.linspace(0, n - 1, T).round().astype(int)
    return (start_f + idx).astype(int)


def remap_path(p):
    """Reescribe el path del NFS al disco local si existe la copia local."""
    if PATH_LOCAL is None:
        return p
    if str(p).startswith(PATH_NFS):
        local = str(p).replace(PATH_NFS, PATH_LOCAL, 1)
        if Path(local).exists():
            return local
    return p


class ClipDataset(Dataset):
    """Lee clips desde video para extracción / entrenamiento end-to-end."""

    def __init__(self, df, T=8, img_size=224):
        self.df       = df
        self.T        = T
        self.img_size = img_size

    def __len__(self): return len(self.df)

    def __getitem__(self, i):
        row     = self.df.iloc[i]
        start_f = int(row["start_frame"])
        end_f   = int(row["end_frame"])
        y       = int(row["y"])
        cap     = cv2.VideoCapture(remap_path(row["path"]))
        ids     = uniform_sample_indices(start_f, end_f, self.T)
        frames, last = [], None
        for fid in ids:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(fid))
            ok, frame = cap.read()
            if not ok:
                frame = last if last is not None else \
                        np.zeros((self.img_size, self.img_size, 3), np.uint8)
            else:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = cv2.resize(frame, (self.img_size, self.img_size),
                                   interpolation=cv2.INTER_LINEAR)
                last  = frame
            frames.append(frame)
        cap.release()
        arr = np.stack(frames).astype(np.float32) / 255.0
        arr = (arr - IMG_MEAN) / IMG_STD
        arr = np.transpose(arr, (3, 0, 1, 2))   # (C, T, H, W)
        return torch.from_numpy(arr), torch.tensor(float(y), dtype=torch.float32)


class EmbeddingDataset(Dataset):
    """Lee embeddings pre-extraídos desde memmap."""

    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self): return len(self.y)

    def __getitem__(self, i):
        x = torch.from_numpy(np.array(self.X[i], dtype=np.float32))
        y = torch.tensor(float(self.y[i]), dtype=torch.float32)
        return x, y


# ─────────────────────────────────────────────────────────────────────────────
# SECCIÓN 3 — Utilidades memmap
# ─────────────────────────────────────────────────────────────────────────────
def create_memmap(path, shape, dtype="float16"):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return np.memmap(path, mode="w+", dtype=dtype, shape=shape)


def load_memmap(path, shape, dtype="float16"):
    return np.memmap(str(path), mode="r", dtype=dtype, shape=shape)


def make_emb_loaders(X_tr, y_tr, X_va, y_va, X_te, y_te, bs=64):
    kw = dict(num_workers=0, pin_memory=False)
    return (
        DataLoader(EmbeddingDataset(X_tr, y_tr), bs, shuffle=True,  **kw),
        DataLoader(EmbeddingDataset(X_va, y_va), bs, shuffle=False, **kw),
        DataLoader(EmbeddingDataset(X_te, y_te), bs, shuffle=False, **kw),
    )


def make_vid_loaders(df_tr, df_va, df_te, bs=8):
    kw = dict(num_workers=NUM_WORKERS, pin_memory=True)
    return (
        DataLoader(ClipDataset(df_tr, T, IMG_SIZE), bs, shuffle=True,  **kw),
        DataLoader(ClipDataset(df_va, T, IMG_SIZE), bs, shuffle=False, **kw),
        DataLoader(ClipDataset(df_te, T, IMG_SIZE), bs, shuffle=False, **kw),
    )


# ─────────────────────────────────────────────────────────────────────────────
# SECCIÓN 4 — LoRA para TimeSformer
# ─────────────────────────────────────────────────────────────────────────────
class LoRALinear(nn.Module):
    """W + (α/r)·B·A — solo A y B son entrenables."""

    def __init__(self, linear: nn.Linear, rank: int,
                 alpha: float, lora_dropout: float = 0.0):
        super().__init__()
        device      = linear.weight.device
        self.weight = linear.weight
        self.bias   = linear.bias
        self.scale  = alpha / rank
        in_f, out_f = linear.in_features, linear.out_features
        self.lora_A  = nn.Parameter(
            torch.randn(rank, in_f, device=device) * 0.01)
        self.lora_B  = nn.Parameter(
            torch.zeros(out_f, rank, device=device))
        self.dropout = nn.Dropout(p=lora_dropout) if lora_dropout > 0 \
                       else nn.Identity()

    def forward(self, x):
        base = F.linear(x, self.weight, self.bias)
        lora = F.linear(self.dropout(x), self.lora_A)
        lora = F.linear(lora, self.lora_B) * self.scale
        return base + lora


def apply_lora(model, rank, alpha, lora_dropout, target_mods=("qkv",)):
    """
    Aplica LoRA sobre las capas Linear cuyo nombre de atributo hoja
    coincida exactamente con alguno de target_mods.
    En HuggingFace TimeSformer: 'qkv' (768→2304, fusiona Q+K+V).
    """
    to_replace = []
    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        if name.split(".")[-1] in target_mods:
            to_replace.append(name)

    for name in to_replace:
        parts  = name.split(".")
        parent = model
        for part in parts[:-1]:
            parent = getattr(parent, part)
        original = getattr(parent, parts[-1])
        setattr(parent, parts[-1],
                LoRALinear(original, rank, alpha, lora_dropout))

    n_total = sum(p.numel() for p in model.parameters())
    n_train = sum(p.numel() for p in model.parameters()
                  if p.requires_grad)
    print(f"  LoRA | capas reemplazadas: {len(to_replace)} | "
          f"entrenables: {n_train:,}/{n_total:,} "
          f"({100 * n_train / n_total:.3f}%)")

    if len(to_replace) == 0:
        raise RuntimeError(
            "LoRA no encontró capas. Verifica los nombres con:\n"
            "  python3 -c \"from transformers import TimesformerModel; "
            "import torch.nn as nn; m = TimesformerModel.from_pretrained("
            "'facebook/timesformer-base-finetuned-k400'); "
            "[print(n) for n,mod in m.named_modules() "
            "if isinstance(mod, nn.Linear)]\""
        )
    return model, n_train, n_total


def load_encoder_frozen():
    """Carga TimeSformer con todos los parámetros congelados."""
    enc = TimesformerModel.from_pretrained(TIMESFORMER_CKPT).to(DEVICE)
    for p in enc.parameters():
        p.requires_grad = False
    enc.eval()
    return enc


def load_encoder_with_lora():
    """Carga TimeSformer, congela todo y aplica LoRA sobre qkv."""
    enc = TimesformerModel.from_pretrained(TIMESFORMER_CKPT).to(DEVICE)
    for p in enc.parameters():
        p.requires_grad = False
    enc, n_lora, n_total = apply_lora(
        enc,
        rank=LORA_CFG["rank"],
        alpha=LORA_CFG["alpha"],
        lora_dropout=LORA_CFG["lora_dropout"],
        target_mods=LORA_CFG["target_mods"],
    )
    return enc, n_lora, n_total


# ─────────────────────────────────────────────────────────────────────────────
# SECCIÓN 5 — Clasificador MLP
# ─────────────────────────────────────────────────────────────────────────────
class AnomalyMLP(nn.Module):
    """fc: d → hidden_dim → 1 con BatchNorm + ReLU + Dropout."""

    def __init__(self, input_dim, hidden_dim, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x): return self.net(x)


def build_mlp(emb_dim=EMB_DIM):
    mlp = AnomalyMLP(emb_dim, MLP_CFG["hidden_dim"],
                     MLP_CFG["dropout"]).to(DEVICE)
    n   = sum(p.numel() for p in mlp.parameters())
    print(f"  MLP | {emb_dim} → {MLP_CFG['hidden_dim']} → 1 | "
          f"dropout={MLP_CFG['dropout']} | params={n:,}")
    return mlp


# ─────────────────────────────────────────────────────────────────────────────
# SECCIÓN 6 — Extracción de embeddings
# ─────────────────────────────────────────────────────────────────────────────
def extract_embeddings(encoder, df, X_mm, y_mm, split_name):
    encoder.eval()
    loader = DataLoader(
        ClipDataset(df, T, IMG_SIZE),
        batch_size=MLP_CFG["batch_vid"],
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )
    print(f"\n  Extrayendo {split_name.upper()} ({len(loader)} batches)...")
    offset = 0
    with torch.no_grad():
        for xb, yb in tqdm(loader, desc=f"  {split_name}", leave=True):
            pv  = xb.to(DEVICE).permute(0, 2, 1, 3, 4).contiguous()
            cls = encoder(pixel_values=pv).last_hidden_state[:, 0, :]
            bs  = cls.shape[0]
            X_mm[offset:offset + bs] = cls.cpu().numpy().astype(X_mm.dtype)
            y_mm[offset:offset + bs] = yb.numpy().astype(y_mm.dtype)
            offset += bs
    X_mm.flush()
    y_mm.flush()
    print(f"  {split_name.upper()} listo: {X_mm.shape}")


# ─────────────────────────────────────────────────────────────────────────────
# SECCIÓN 7 — Training loops
# ─────────────────────────────────────────────────────────────────────────────
def compute_metrics(probs, labels, thr=0.5):
    preds = (probs >= thr).astype(int)
    fp    = int(((preds == 1) & (labels == 0)).sum())
    tn    = int(((preds == 0) & (labels == 0)).sum())
    far   = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    return {
        "AUC":       float(roc_auc_score(labels, probs)),
        "Accuracy":  float(accuracy_score(labels, preds)),
        "Precision": float(precision_score(labels, preds, zero_division=0)),
        "Recall":    float(recall_score(labels, preds, zero_division=0)),
        "F1":        float(f1_score(labels, preds, zero_division=0)),
        "FAR":       float(far),
    }


def _print_header(label, n_train, lr, epochs, patience):
    print(f"\n  [{label}] Parámetros entrenables: {n_train:,}")
    print(f"  LR={lr} | epochs={epochs} | patience={patience}")
    print(f"  {'Ep':>4} | {'TrainLoss':>10} | {'ValAUC':>8} | "
          f"{'No↑':>4} | {'Time':>8}")
    print("  " + "─" * 52)


def train_on_embeddings(mlp, tr_ldr, va_ldr, lr, epochs, patience,
                        ckpt_path, label):
    """Entrena el MLP sobre embeddings pre-extraídos."""
    crit      = nn.BCEWithLogitsLoss()
    opt       = torch.optim.Adam(mlp.parameters(), lr=lr)
    best_auc  = 0.0
    no_imp    = 0
    history   = {"train_loss": [], "val_loss": [], "val_auc": []}
    t0_total  = time.time()
    n_train   = sum(p.numel() for p in mlp.parameters())
    _print_header(label, n_train, lr, epochs, patience)

    for ep in range(1, epochs + 1):
        t0 = time.time()
        # — train —
        mlp.train()
        tr_loss = 0.0
        for x, y in tqdm(tr_ldr, desc="  train", leave=False):
            x, y = x.to(DEVICE), y.to(DEVICE)
            opt.zero_grad()
            loss = crit(mlp(x).squeeze(1), y)
            loss.backward()
            opt.step()
            tr_loss += loss.item() * len(y)
        tr_loss /= len(tr_ldr.dataset)

        # — val —
        mlp.eval()
        va_loss, all_p, all_l = 0.0, [], []
        with torch.no_grad():
            for x, y in tqdm(va_ldr, desc="  val  ", leave=False):
                x, y   = x.to(DEVICE), y.to(DEVICE)
                logits = mlp(x).squeeze(1)
                va_loss += crit(logits, y).item() * len(y)
                all_p.append(torch.sigmoid(logits).cpu().numpy())
                all_l.append(y.cpu().numpy())
        va_loss /= len(va_ldr.dataset)
        va_auc   = roc_auc_score(np.concatenate(all_l),
                                 np.concatenate(all_p))

        history["train_loss"].append(tr_loss)
        history["val_loss"].append(va_loss)
        history["val_auc"].append(va_auc)

        if va_auc > best_auc:
            best_auc, no_imp = va_auc, 0
            torch.save(mlp.state_dict(), ckpt_path)
            flag = "✓ best"
        else:
            no_imp += 1
            flag = ""

        print(f"  {ep:>4} | {tr_loss:>10.4f} | {va_auc:>8.4f} | "
              f"{no_imp:>4} | {time.time()-t0:>7.1f}s | {flag}")

        if no_imp >= patience:
            print(f"\n  ⏹  Early stopping en época {ep}")
            break

    history["best_auc"]   = best_auc
    history["total_time"] = time.time() - t0_total
    history["epochs_run"] = ep
    print(f"\n  ✔ Mejor AUC val: {best_auc:.4f} | "
          f"Tiempo: {history['total_time']/60:.1f} min")
    return history


class FullPipeline(nn.Module):
    """Encoder + MLP — forward sobre video crudo (C,T,H,W)."""

    def __init__(self, encoder, mlp):
        super().__init__()
        self.encoder = encoder
        self.mlp     = mlp

    def forward(self, x):
        pv  = x.permute(0, 2, 1, 3, 4).contiguous()
        cls = self.encoder(pixel_values=pv).last_hidden_state[:, 0, :]
        return self.mlp(cls)


def train_endtoend(full_model, tr_ldr, va_ldr, lr, epochs, patience,
                   ckpt_path, label):
    """Entrena LoRA+MLP end-to-end sobre video crudo."""
    crit     = nn.BCEWithLogitsLoss()
    opt      = torch.optim.Adam(
        [p for p in full_model.parameters() if p.requires_grad], lr=lr)
    best_auc = 0.0
    no_imp   = 0
    history  = {"train_loss": [], "val_auc": []}
    t0_total = time.time()
    n_train  = sum(p.numel() for p in full_model.parameters()
                   if p.requires_grad)
    _print_header(label, n_train, lr, epochs, patience)

    for ep in range(1, epochs + 1):
        t0 = time.time()
        # — train —
        full_model.train()
        tr_loss = 0.0
        for xb, yb in tqdm(tr_ldr, desc="  train", leave=False):
            xb = xb.to(DEVICE)
            yb = yb.to(DEVICE)
            opt.zero_grad()
            loss = crit(full_model(xb).squeeze(1), yb)
            loss.backward()
            opt.step()
            tr_loss += loss.item() * len(yb)
        tr_loss /= len(tr_ldr.dataset)

        # — val —
        full_model.eval()
        all_p, all_l = [], []
        with torch.no_grad():
            for xb, yb in tqdm(va_ldr, desc="  val  ", leave=False):
                out = full_model(xb.to(DEVICE)).squeeze(1)
                all_p.append(torch.sigmoid(out).cpu().numpy())
                all_l.append(yb.numpy())
        va_auc = roc_auc_score(np.concatenate(all_l),
                               np.concatenate(all_p))

        history["train_loss"].append(tr_loss)
        history["val_auc"].append(va_auc)

        if va_auc > best_auc:
            best_auc, no_imp = va_auc, 0
            torch.save(full_model.state_dict(), ckpt_path)
            flag = "✓ best"
        else:
            no_imp += 1
            flag = ""

        print(f"  {ep:>4} | {tr_loss:>10.4f} | {va_auc:>8.4f} | "
              f"{no_imp:>4} | {time.time()-t0:>7.1f}s | {flag}")

        if no_imp >= patience:
            print(f"\n  ⏹  Early stopping en época {ep}")
            break

    history["best_auc"]   = best_auc
    history["total_time"] = time.time() - t0_total
    history["epochs_run"] = ep
    print(f"\n  ✔ Mejor AUC val: {best_auc:.4f} | "
          f"Tiempo: {history['total_time']/60:.1f} min")
    return history


# ─────────────────────────────────────────────────────────────────────────────
# SECCIÓN 8 — Métricas operacionales
# ─────────────────────────────────────────────────────────────────────────────
def measure_operational_metrics(encoder, mlp, n_clips=100):
    encoder.eval()
    mlp.eval()
    dummy = torch.randn(1, 3, T, IMG_SIZE, IMG_SIZE, device=DEVICE)
    print("\n  Midiendo métricas operacionales (warmup 10 clips)...")
    for _ in range(10):
        with torch.no_grad():
            pv  = dummy.permute(0, 2, 1, 3, 4).contiguous()
            cls = encoder(pixel_values=pv).last_hidden_state[:, 0, :]
            mlp(cls)
    if DEVICE.type == "cuda":
        torch.cuda.synchronize()
    times = []
    for _ in tqdm(range(n_clips), desc="  latencia", leave=False):
        t0 = time.perf_counter()
        with torch.no_grad():
            pv  = dummy.permute(0, 2, 1, 3, 4).contiguous()
            cls = encoder(pixel_values=pv).last_hidden_state[:, 0, :]
            mlp(cls)
        if DEVICE.type == "cuda":
            torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
    ms  = float(np.mean(times) * 1000)
    fps = float(1.0 / np.mean(times))
    res = {"latencia_ms_clip": round(ms, 2),
           "fps": round(fps, 1),
           "viable_realtime": fps >= 30.0}
    print(f"  FPS: {fps:.1f} | ms/clip: {ms:.2f} | "
          f"Tiempo real: {'✔' if fps >= 30 else '✗'}")
    return res


# ─────────────────────────────────────────────────────────────────────────────
# SECCIÓN 9 — Evaluación test
# ─────────────────────────────────────────────────────────────────────────────
@torch.no_grad()
def eval_test_emb(mlp, te_ldr, label):
    mlp.eval()
    all_p, all_l = [], []
    for x, y in tqdm(te_ldr, desc=f"  test [{label}]", leave=False):
        logits = mlp(x.to(DEVICE)).squeeze(1)
        all_p.append(torch.sigmoid(logits).cpu().numpy())
        all_l.append(y.numpy())
    return np.concatenate(all_p), np.concatenate(all_l)


@torch.no_grad()
def eval_test_video(full_model, te_ldr, label):
    full_model.eval()
    all_p, all_l = [], []
    for x, y in tqdm(te_ldr, desc=f"  test [{label}]", leave=False):
        out = full_model(x.to(DEVICE)).squeeze(1)
        all_p.append(torch.sigmoid(out).cpu().numpy())
        all_l.append(y.numpy())
    return np.concatenate(all_p), np.concatenate(all_l)


# ─────────────────────────────────────────────────────────────────────────────
# SECCIÓN 10 — Gráficos
# ─────────────────────────────────────────────────────────────────────────────
COLORS = {"BASELINE": "steelblue", "LORA+MLP": "tomato",
          "SOLO_LORA": "seagreen"}


def plot_training_curves(histories):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("TimeSformer — Curvas de entrenamiento",
                 fontsize=14, fontweight="bold")
    for cfg, hist in histories.items():
        c = COLORS.get(cfg, "gray")
        if "train_loss" in hist:
            axes[0].plot(hist["train_loss"], "--", color=c,
                         label=f"{cfg} — Train", alpha=0.7)
        if "val_loss" in hist:
            axes[0].plot(hist["val_loss"], "-", color=c,
                         label=f"{cfg} — Val")
        if "val_auc" in hist:
            axes[1].plot(hist["val_auc"], "-", color=c, label=cfg)
            axes[1].axhline(hist["best_auc"], linestyle=":",
                            color=c, alpha=0.4)
    axes[0].set(xlabel="Época", ylabel="BCE Loss",
                title="Curvas de pérdida")
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.3)
    axes[1].set(xlabel="Época", ylabel="AUC (validación)",
                title="AUC por época", ylim=(0.5, 1.0))
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.3)
    plt.tight_layout()
    p = FIG_DIR / "ts_01_curvas_entrenamiento.png"
    plt.savefig(p, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Guardado: {p}")


def plot_roc(test_results):
    plt.figure(figsize=(7, 6))
    for cfg, (probs, labels, _) in test_results.items():
        fpr, tpr, _ = roc_curve(labels, probs)
        auc = roc_auc_score(labels, probs)
        plt.plot(fpr, tpr, color=COLORS.get(cfg, "gray"),
                 label=f"{cfg} (AUC={auc:.4f})")
    plt.plot([0, 1], [0, 1], "k--", label="Aleatorio")
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.title("Curva ROC — TimeSformer (Test)")
    plt.legend(fontsize=9)
    plt.grid(alpha=0.3)
    p = FIG_DIR / "ts_02_curva_roc.png"
    plt.savefig(p, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Guardado: {p}")


def plot_metrics(test_results):
    keys = ["AUC", "F1", "Precision", "Recall", "FAR"]
    cfgs = list(test_results.keys())
    x    = np.arange(len(keys))
    w    = 0.25
    fig, ax = plt.subplots(figsize=(13, 5))
    for i, cfg in enumerate(cfgs):
        vals = [test_results[cfg][2][k] for k in keys]
        bars = ax.bar(x + i * w, vals, w, label=cfg,
                      color=COLORS.get(cfg, "gray"), alpha=0.85)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.005,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=7)
    ax.set_xticks(x + w)
    ax.set_xticklabels(keys)
    ax.set_ylabel("Valor")
    ax.set_title("Comparación de métricas — TimeSformer (Test)")
    ax.legend()
    ax.set_ylim(0, 1.12)
    ax.axhline(1.0, linestyle="--", color="gray", alpha=0.4)
    ax.text(len(keys) - 0.3, 0.02, "FAR: menor es mejor",
            fontsize=7, color="gray", ha="right")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    p = FIG_DIR / "ts_03_comparacion_metricas.png"
    plt.savefig(p, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Guardado: {p}")


def plot_confusion(test_results):
    cfgs = list(test_results.keys())
    fig, axes = plt.subplots(1, len(cfgs), figsize=(6 * len(cfgs), 5))
    if len(cfgs) == 1:
        axes = [axes]
    fig.suptitle("TimeSformer — Matrices de confusión (Test)",
                 fontsize=13, fontweight="bold")
    for ax, cfg in zip(axes, cfgs):
        probs, labels, _ = test_results[cfg]
        cm = confusion_matrix(labels, (probs >= 0.5).astype(int))
        ax.imshow(cm, cmap="Blues")
        ax.set_title(f"CM — {cfg}", fontweight="bold")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(["Normal", "Anómalo"])
        ax.set_yticklabels(["Normal", "Anómalo"])
        for i in range(2):
            for j in range(2):
                ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                        color="white" if cm[i, j] > cm.max() / 2 else "navy",
                        fontsize=13, fontweight="bold")
    plt.tight_layout()
    p = FIG_DIR / "ts_04_matrices_confusion.png"
    plt.savefig(p, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Guardado: {p}")


def plot_scores(test_results):
    cfgs = list(test_results.keys())
    fig, axes = plt.subplots(1, len(cfgs), figsize=(6 * len(cfgs), 4))
    if len(cfgs) == 1:
        axes = [axes]
    fig.suptitle("TimeSformer — Distribución de scores (Test)",
                 fontsize=13, fontweight="bold")
    for ax, cfg in zip(axes, cfgs):
        probs, labels, _ = test_results[cfg]
        ax.hist(probs[labels == 0], bins=50, alpha=0.6,
                color="steelblue", label="Normal", density=True)
        ax.hist(probs[labels == 1], bins=50, alpha=0.6,
                color="tomato", label="Anómalo", density=True)
        ax.axvline(0.5, linestyle="--", color="black", linewidth=1)
        ax.set_title(f"Scores — {cfg}")
        ax.set_xlabel("Score de anomalía")
        ax.set_ylabel("Densidad")
        ax.legend(fontsize=8)
    plt.tight_layout()
    p = FIG_DIR / "ts_05_distribucion_scores.png"
    plt.savefig(p, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Guardado: {p}")


# ─────────────────────────────────────────────────────────────────────────────
# SECCIÓN 11 — Orquestador principal
# ─────────────────────────────────────────────────────────────────────────────
def run_pipeline():

    # ── Checkpoints de fase ───────────────────────────────────────────────────
    ckpt_emb       = OUT_DIR / ".phase_emb_done"
    ckpt_baseline  = OUT_DIR / ".phase_baseline_done"
    ckpt_lora      = OUT_DIR / ".phase_lora_done"
    ckpt_solo_lora = OUT_DIR / ".phase_solo_lora_done"

    # ── Paths memmap ──────────────────────────────────────────────────────────
    paths = {
        "X_tr": OUT_DIR / "ts_X_train.mmap",
        "y_tr": OUT_DIR / "ts_y_train.mmap",
        "X_va": OUT_DIR / "ts_X_val.mmap",
        "y_va": OUT_DIR / "ts_y_val.mmap",
        "X_te": OUT_DIR / "ts_X_test.mmap",
        "y_te": OUT_DIR / "ts_y_test.mmap",
    }

    # ══════════════════════════════════════════════════════════════════════════
    # FASE 1 — Extracción de embeddings (TimeSformer congelado)
    # ══════════════════════════════════════════════════════════════════════════
    if ckpt_emb.exists():
        print("\n  [SKIP] Embeddings ya extraídos, cargando...")
        X_tr = load_memmap(paths["X_tr"], (len(df_train), EMB_DIM))
        y_tr = load_memmap(paths["y_tr"], (len(df_train),), "int8")
        X_va = load_memmap(paths["X_va"], (len(df_val),   EMB_DIM))
        y_va = load_memmap(paths["y_va"], (len(df_val),),   "int8")
        X_te = load_memmap(paths["X_te"], (len(df_test),  EMB_DIM))
        y_te = load_memmap(paths["y_te"], (len(df_test),),  "int8")
    else:
        print("\n" + "=" * 68)
        print("  FASE 1 — EXTRACCIÓN EMBEDDINGS (TimeSformer congelado)")
        print("=" * 68)
        enc = load_encoder_frozen()
        X_tr = create_memmap(paths["X_tr"], (len(df_train), EMB_DIM))
        y_tr = create_memmap(paths["y_tr"], (len(df_train),), "int8")
        X_va = create_memmap(paths["X_va"], (len(df_val),   EMB_DIM))
        y_va = create_memmap(paths["y_va"], (len(df_val),),   "int8")
        X_te = create_memmap(paths["X_te"], (len(df_test),  EMB_DIM))
        y_te = create_memmap(paths["y_te"], (len(df_test),),  "int8")
        extract_embeddings(enc, df_train, X_tr, y_tr, "train")
        extract_embeddings(enc, df_val,   X_va, y_va, "val")
        extract_embeddings(enc, df_test,  X_te, y_te, "test")
        del enc
        if DEVICE.type == "cuda":
            torch.cuda.empty_cache()
        ckpt_emb.touch()
        print("  ✔ Fase 1 completada.")

    tr_emb, va_emb, te_emb = make_emb_loaders(
        X_tr, y_tr, X_va, y_va, X_te, y_te, MLP_CFG["batch_size"])

    # ══════════════════════════════════════════════════════════════════════════
    # FASE 2 — BASELINE: TimeSformer congelado + MLP
    # ══════════════════════════════════════════════════════════════════════════
    ckpt_base_pth  = OUT_DIR / "ts_baseline_best.pth"
    baseline_json  = OUT_DIR / "ts_baseline_result.json"

    if ckpt_baseline.exists() and baseline_json.exists():
        print("\n  [SKIP] Baseline ya entrenado.")
        hist_base = json.loads(baseline_json.read_text())
    else:
        print("\n" + "=" * 68)
        print("  FASE 2 — BASELINE: TimeSformer congelado + MLP")
        print("=" * 68)
        set_seed(MLP_CFG["seed"])
        mlp = build_mlp()
        hist_base = train_on_embeddings(
            mlp, tr_emb, va_emb,
            lr=MLP_CFG["lr"],
            epochs=MLP_CFG["epochs"],
            patience=MLP_CFG["patience"],
            ckpt_path=ckpt_base_pth,
            label="BASELINE",
        )
        baseline_json.write_text(json.dumps(hist_base, indent=2))
        ckpt_baseline.touch()
        del mlp
        if DEVICE.type == "cuda":
            torch.cuda.empty_cache()
        print("  ✔ Fase 2 completada.")

    # ══════════════════════════════════════════════════════════════════════════
    # FASE 3 — LORA+MLP: end-to-end sobre video crudo
    # ══════════════════════════════════════════════════════════════════════════
    ckpt_lora_pth = OUT_DIR / "ts_lora_best.pth"
    lora_json     = OUT_DIR / "ts_lora_result.json"

    if ckpt_lora.exists() and lora_json.exists():
        print("\n  [SKIP] LoRA+MLP ya entrenado.")
        hist_lora = json.loads(lora_json.read_text())
    else:
        print("\n" + "=" * 68)
        print("  FASE 3 — LORA+MLP: TimeSformer+LoRA end-to-end")
        print(f"  r={LORA_CFG['rank']} | α={LORA_CFG['alpha']} | "
              f"dropout={LORA_CFG['lora_dropout']} | "
              f"lr={MLP_CFG['lr_e2e']} (reducido para Transformer)")
        print("=" * 68)
        set_seed(MLP_CFG["seed"])
        enc_lora, n_lora, n_total = load_encoder_with_lora()
        mlp_lora = build_mlp()
        full     = FullPipeline(enc_lora, mlp_lora).to(DEVICE)
        tr_vid, va_vid, _ = make_vid_loaders(
            df_train, df_val, df_test, MLP_CFG["batch_vid"])
        hist_lora = train_endtoend(
            full, tr_vid, va_vid,
            lr=MLP_CFG["lr_e2e"],
            epochs=MLP_CFG["epochs"],
            patience=MLP_CFG["patience"],
            ckpt_path=ckpt_lora_pth,
            label="LORA+MLP",
        )
        hist_lora["lora_params"]  = n_lora
        hist_lora["total_params"] = n_total
        hist_lora["lora_pct"]     = round(100 * n_lora / n_total, 3)
        lora_json.write_text(json.dumps(hist_lora, indent=2))
        ckpt_lora.touch()
        del full, enc_lora, mlp_lora
        if DEVICE.type == "cuda":
            torch.cuda.empty_cache()
        print("  ✔ Fase 3 completada.")

    # ══════════════════════════════════════════════════════════════════════════
    # FASE 4 — SOLO_LORA: MLP congelado del baseline + solo LoRA aprende
    # ══════════════════════════════════════════════════════════════════════════
    ckpt_sl_pth  = OUT_DIR / "ts_solo_lora_best.pth"
    solo_lora_json = OUT_DIR / "ts_solo_lora_result.json"

    if ckpt_solo_lora.exists() and solo_lora_json.exists():
        print("\n  [SKIP] Solo LoRA ya entrenado.")
        hist_solo = json.loads(solo_lora_json.read_text())
    else:
        print("\n" + "=" * 68)
        print("  FASE 4 — SOLO_LORA: MLP baseline congelado + solo LoRA")
        print(f"  El MLP ya sabe clasificar (cargado desde baseline).")
        print(f"  Solo las matrices A y B de LoRA aprenderán.")
        print("=" * 68)
        set_seed(MLP_CFG["seed"])

        # Cargar encoder con LoRA
        enc_sl, n_lora, n_total = load_encoder_with_lora()

        # Cargar MLP desde checkpoint baseline y CONGELARLO
        mlp_sl = build_mlp()
        mlp_sl.load_state_dict(torch.load(ckpt_base_pth, map_location=DEVICE))
        for p in mlp_sl.parameters():
            p.requires_grad = False
        print(f"  MLP cargado desde baseline y congelado.")

        full_sl = FullPipeline(enc_sl, mlp_sl).to(DEVICE)

        tr_vid, va_vid, _ = make_vid_loaders(
            df_train, df_val, df_test, MLP_CFG["batch_vid"])

        hist_solo = train_endtoend(
            full_sl, tr_vid, va_vid,
            lr=MLP_CFG["lr_e2e"],
            epochs=MLP_CFG["epochs"],
            patience=MLP_CFG["patience"],
            ckpt_path=ckpt_sl_pth,
            label="SOLO_LORA",
        )
        hist_solo["lora_params"]  = n_lora
        hist_solo["total_params"] = n_total
        hist_solo["lora_pct"]     = round(100 * n_lora / n_total, 3)
        solo_lora_json.write_text(json.dumps(hist_solo, indent=2))
        ckpt_solo_lora.touch()
        del full_sl, enc_sl, mlp_sl
        if DEVICE.type == "cuda":
            torch.cuda.empty_cache()
        print("  ✔ Fase 4 completada.")

    # ══════════════════════════════════════════════════════════════════════════
    # FASE 5 — Evaluación final sobre test
    # ══════════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 68)
    print("  FASE 5 — EVALUACIÓN FINAL SOBRE TEST")
    print("=" * 68)
    set_seed(MLP_CFG["seed"])

    # — Baseline —
    mlp_eval = build_mlp()
    mlp_eval.load_state_dict(torch.load(ckpt_base_pth, map_location=DEVICE))
    p_base, l_base = eval_test_emb(mlp_eval, te_emb, "BASELINE")
    m_base = compute_metrics(p_base, l_base)
    del mlp_eval

    # — LoRA+MLP —
    enc_eval, _, _ = load_encoder_with_lora()
    mlp_eval       = build_mlp()
    full_eval      = FullPipeline(enc_eval, mlp_eval).to(DEVICE)
    full_eval.load_state_dict(torch.load(ckpt_lora_pth, map_location=DEVICE))
    _, _, te_vid = make_vid_loaders(df_train, df_val, df_test,
                                    MLP_CFG["batch_vid"])
    p_lora, l_lora = eval_test_video(full_eval, te_vid, "LORA+MLP")
    m_lora = compute_metrics(p_lora, l_lora)

    # Métricas operacionales con LoRA+MLP
    op = measure_operational_metrics(enc_eval, mlp_eval)
    del full_eval

    # — Solo LoRA —
    enc_sl, _, _ = load_encoder_with_lora()
    mlp_sl       = build_mlp()
    full_sl      = FullPipeline(enc_sl, mlp_sl).to(DEVICE)
    full_sl.load_state_dict(torch.load(ckpt_sl_pth, map_location=DEVICE))
    _, _, te_vid2 = make_vid_loaders(df_train, df_val, df_test,
                                     MLP_CFG["batch_vid"])
    p_solo, l_solo = eval_test_video(full_sl, te_vid2, "SOLO_LORA")
    m_solo = compute_metrics(p_solo, l_solo)
    del full_sl

    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()

    # ══════════════════════════════════════════════════════════════════════════
    # FASE 6 — Guardar resultados y gráficos
    # ══════════════════════════════════════════════════════════════════════════
    final = {
        "codificador":        "TimeSformer",
        "checkpoint":         TIMESFORMER_CKPT,
        "config_mlp":         MLP_CFG,
        "config_lora":        LORA_CFG,
        "BASELINE":  {"test_metrics": m_base,
                      "best_val_auc": hist_base["best_auc"],
                      "epochs_run":   hist_base.get("epochs_run")},
        "LORA+MLP":  {"test_metrics": m_lora,
                      "best_val_auc": hist_lora["best_auc"],
                      "epochs_run":   hist_lora.get("epochs_run"),
                      "lora_params":  hist_lora.get("lora_params"),
                      "lora_pct":     hist_lora.get("lora_pct")},
        "SOLO_LORA": {"test_metrics": m_solo,
                      "best_val_auc": hist_solo["best_auc"],
                      "epochs_run":   hist_solo.get("epochs_run"),
                      "lora_params":  hist_solo.get("lora_params"),
                      "lora_pct":     hist_solo.get("lora_pct")},
        "operational_metrics": op,
    }
    (OUT_DIR / "ts_results.json").write_text(json.dumps(final, indent=2))

    # Tabla resumen
    print(f"\n  {'Config':<12} {'AUC':>6} {'F1':>6} "
          f"{'Prec':>6} {'Rec':>6} {'FAR':>6}")
    print("  " + "─" * 48)
    for cfg, m in [("BASELINE", m_base), ("LORA+MLP", m_lora),
                   ("SOLO_LORA", m_solo)]:
        print(f"  {cfg:<12} {m['AUC']:>6.4f} {m['F1']:>6.4f} "
              f"{m['Precision']:>6.4f} {m['Recall']:>6.4f} "
              f"{m['FAR']:>6.4f}")
    print(f"\n  FPS: {op['fps']:.1f} | ms/clip: {op['latencia_ms_clip']:.2f}")

    # Gráficos
    print("\n  Generando gráficos...")
    histories = {
        "BASELINE":  hist_base,
        "LORA+MLP":  hist_lora,
        "SOLO_LORA": hist_solo,
    }
    test_results = {
        "BASELINE":  (p_base, l_base, m_base),
        "LORA+MLP":  (p_lora, l_lora, m_lora),
        "SOLO_LORA": (p_solo, l_solo, m_solo),
    }
    plot_training_curves(histories)
    plot_roc(test_results)
    plot_metrics(test_results)
    plot_confusion(test_results)
    plot_scores(test_results)

    print(f"\n  ✅ Pipeline completado.")
    print(f"  Resultados : {OUT_DIR}/ts_results.json")
    print(f"  Gráficos   : {FIG_DIR}/")
    return final


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    t0 = time.time()
    print("=" * 68)
    print("  TimeSformer — 3 experimentos (igual que I3D)")
    print(f"  MLP : fc | hidden=128 | dropout=0.3 | lr=1e-3")
    print(f"  LoRA: r={LORA_CFG['rank']} | α={LORA_CFG['alpha']} | "
          f"dropout={LORA_CFG['lora_dropout']} | "
          f"target={','.join(LORA_CFG['target_mods'])} | lr=1e-4")
    print("=" * 68)
    results = run_pipeline()
    total = time.time() - t0
    print(f"\n  Tiempo total: {total/3600:.2f} h ({total/60:.1f} min)")
