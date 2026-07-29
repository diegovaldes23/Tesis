#!/usr/bin/env python3
"""
=============================================================================
  FASE 4 — SOLO_LORA  (script independiente)
=============================================================================
  Entrena solo los parámetros LoRA (FFN fc) con el MLP del baseline congelado.
  Requiere que Fase 2 (BASELINE) esté completada:
    processed/realswin_results/realswin_baseline_best.pth

  Guarda resultados compatibles con el pipeline principal (videoswin_real.py)
  para que Fase 5 (evaluación) pueda correr normalmente después.

Ejecutar en GPU 1 (en paralelo con fase3_lora_mlp.py en GPU 0):
    screen -S fase4
    CUDA_VISIBLE_DEVICES=1 python3 fase4_solo_lora.py 2>&1 | tee fase4.log
    Ctrl+A D  →  tail -f fase4.log
=============================================================================
"""
import json, time, warnings
from pathlib import Path
warnings.filterwarnings("ignore")

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
from torchvision.models.video import swin3d_b, Swin3D_B_Weights

# ─────────────────────────────────────────────────────────────────────────────
# Configuración
# ─────────────────────────────────────────────────────────────────────────────
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
INDEX_PATH  = Path("processed/index_clips.csv")
OUT_DIR     = Path("processed/realswin_results")
OUT_DIR.mkdir(parents=True, exist_ok=True)

CKPT_BASE   = OUT_DIR / "realswin_baseline_best.pth"   # requerido
CKPT_PATH   = OUT_DIR / "realswin_solo_lora_best.pth"
RESULT_JSON = OUT_DIR / "realswin_solo_lora_result.json"
PHASE_DONE  = OUT_DIR / ".phase_solo_lora_done"

T           = 32
IMG_SIZE    = 224
NUM_WORKERS = 4
IMG_MEAN    = np.array([0.4850, 0.4560, 0.4060], dtype=np.float32)
IMG_STD     = np.array([0.2290, 0.2240, 0.2250], dtype=np.float32)
EMB_DIM     = 1024
SEED        = 42

EPOCHS      = 50
PATIENCE    = 8
LR          = 1e-4
BATCH_VID   = 4
HIDDEN_DIM  = 128
DROPOUT_MLP = 0.3

LORA_RANK    = 4
LORA_ALPHA   = 8
LORA_DROPOUT = 0.10
LORA_TARGETS = ("0", "3")   # fc1, fc2 del FFN — verificado con diagnostico_lora.py


def set_seed(s):
    torch.manual_seed(s)
    np.random.seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)


# ─────────────────────────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────────────────────────
def uniform_sample_indices(start_f, end_f, T):
    n   = max(1, end_f - start_f)
    idx = np.linspace(0, n - 1, T).round().astype(int)
    return (start_f + idx).astype(int)


class ClipDataset(Dataset):
    def __init__(self, df):
        self.df = df

    def __len__(self): return len(self.df)

    def __getitem__(self, i):
        row = self.df.iloc[i]
        cap = cv2.VideoCapture(row["path"])
        ids = uniform_sample_indices(int(row["start_frame"]),
                                     int(row["end_frame"]), T)
        frames, last = [], None
        for fid in ids:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(fid))
            ok, frame = cap.read()
            if not ok:
                frame = last if last is not None else \
                        np.zeros((IMG_SIZE, IMG_SIZE, 3), np.uint8)
            else:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
                last  = frame
            frames.append(frame)
        cap.release()
        arr = np.stack(frames).astype(np.float32) / 255.0
        arr = (arr - IMG_MEAN) / IMG_STD
        arr = np.transpose(arr, (3, 0, 1, 2))
        return (torch.from_numpy(arr),
                torch.tensor(float(row["y"]), dtype=torch.float32))


# ─────────────────────────────────────────────────────────────────────────────
# LoRALinear
# ─────────────────────────────────────────────────────────────────────────────
class LoRALinear(nn.Module):
    def __init__(self, linear, rank, alpha, lora_dropout=0.0):
        super().__init__()
        self.register_buffer("base_weight", linear.weight.data.clone())
        if linear.bias is not None:
            self.register_buffer("base_bias", linear.bias.data.clone())
        else:
            self.register_buffer("base_bias", None)
        self.scale  = alpha / rank
        dev = linear.weight.device
        self.lora_A = nn.Parameter(
            torch.randn(rank, linear.in_features, device=dev) * 0.01)
        self.lora_B = nn.Parameter(
            torch.zeros(linear.out_features, rank, device=dev))
        self.dropout = nn.Dropout(p=lora_dropout) if lora_dropout > 0 \
                       else nn.Identity()

    @property
    def weight(self): return self.base_weight
    @property
    def bias(self):   return self.base_bias

    def forward(self, x):
        base = F.linear(x, self.base_weight, self.base_bias)
        lora = F.linear(self.dropout(x), self.lora_A)
        lora = F.linear(lora, self.lora_B) * self.scale
        return base + lora


# ─────────────────────────────────────────────────────────────────────────────
# Modelos
# ─────────────────────────────────────────────────────────────────────────────
def build_encoder_lora():
    weights = Swin3D_B_Weights.KINETICS400_IMAGENET22K_V1
    model   = swin3d_b(weights=weights)
    model.head = nn.Identity()
    for p in model.parameters():
        p.requires_grad = False

    replaced = 0
    for name, module in list(model.named_modules()):
        if not isinstance(module, nn.Linear): continue
        leaf = name.split(".")[-1]
        if leaf in LORA_TARGETS and ".mlp." in name:
            parts  = name.split(".")
            parent = model
            for part in parts[:-1]: parent = getattr(parent, part)
            orig = getattr(parent, parts[-1])
            setattr(parent, parts[-1],
                    LoRALinear(orig, LORA_RANK, LORA_ALPHA, LORA_DROPOUT))
            replaced += 1

    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    print(f"  Swin3D+LoRA | capas: {replaced} | "
          f"entrenables: {n_train:,}/{n_total:,} "
          f"({100*n_train/n_total:.3f}%)")
    model.eval()
    return model.to(DEVICE), n_train, n_total


class AnomalyMLP(nn.Module):
    """Misma arquitectura que el pipeline principal (claves net.X)."""
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(EMB_DIM, HIDDEN_DIM),
            nn.BatchNorm1d(HIDDEN_DIM),
            nn.ReLU(inplace=True),
            nn.Dropout(DROPOUT_MLP),
            nn.Linear(HIDDEN_DIM, 1),
        )
    def forward(self, x): return self.net(x)


def build_mlp_frozen():
    """Carga MLP desde baseline y lo congela."""
    assert CKPT_BASE.exists(), (
        f"No se encuentra el checkpoint baseline: {CKPT_BASE}\n"
        "Asegúrate de que la Fase 2 esté completada antes de correr Fase 4.")
    mlp = AnomalyMLP().to(DEVICE)
    mlp.load_state_dict(torch.load(CKPT_BASE, map_location=DEVICE))
    for p in mlp.parameters():
        p.requires_grad = False
    n = sum(p.numel() for p in mlp.parameters())
    print(f"  MLP | {EMB_DIM}→{HIDDEN_DIM}→1 | params={n:,} | CONGELADO ✓")
    return mlp


class FullPipeline(nn.Module):
    def __init__(self, encoder, mlp):
        super().__init__()
        self.encoder = encoder
        self.mlp     = mlp

    def forward(self, x):
        return self.mlp(self.encoder(x))


# ─────────────────────────────────────────────────────────────────────────────
# Training loop SOLO_LORA
# ─────────────────────────────────────────────────────────────────────────────
def train_solo_lora(full_model, tr_ldr, va_ldr):
    crit     = nn.BCEWithLogitsLoss()
    opt      = torch.optim.Adam(
        [p for p in full_model.parameters() if p.requires_grad], lr=LR)
    best_auc = 0.0
    no_imp   = 0
    history  = {"train_loss": [], "val_auc": []}
    t0_total = time.time()
    n_train  = sum(p.numel() for p in full_model.parameters()
                   if p.requires_grad)

    print(f"\n  [SOLO_LORA] Params entrenables: {n_train:,} (solo LoRA)")
    print(f"  LR={LR} | epochs={EPOCHS} | patience={PATIENCE}")
    print(f"  {'Ep':>4} | {'TrainLoss':>10} | {'ValAUC':>8} | "
          f"{'No↑':>4} | {'Time':>9}")
    print("  " + "─" * 52)

    for ep in range(1, EPOCHS + 1):
        t0 = time.time()
        # Encoder en train (LoRA activo), MLP en eval (BatchNorm fijo)
        full_model.encoder.train()
        full_model.mlp.eval()
        tr_loss = 0.0

        for xb, yb in tqdm(tr_ldr, desc="  train", leave=False):
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad()
            out  = full_model(xb).squeeze(1)

            # Verificación en primer batch de primera época
            if ep == 1 and tr_loss == 0.0 and out.grad_fn is None:
                raise RuntimeError(
                    "[SOLO_LORA] out no tiene grad_fn — "
                    "LoRA no está contribuyendo al grafo. "
                    "Corre diagnostico_lora.py para verificar.")

            loss = crit(out, yb)
            loss.backward()
            opt.step()
            tr_loss += loss.item() * len(yb)
        tr_loss /= len(tr_ldr.dataset)

        full_model.eval()
        all_p, all_l = [], []
        with torch.no_grad():
            for xb, yb in tqdm(va_ldr, desc="  val  ", leave=False):
                out = full_model(xb.to(DEVICE)).squeeze(1)
                all_p.append(torch.sigmoid(out).cpu().numpy())
                all_l.append(yb.numpy())
        va_auc = roc_auc_score(np.concatenate(all_l), np.concatenate(all_p))

        history["train_loss"].append(tr_loss)
        history["val_auc"].append(va_auc)

        if va_auc > best_auc:
            best_auc, no_imp = va_auc, 0
            torch.save(full_model.state_dict(), CKPT_PATH)
            flag = "✓ best"
        else:
            no_imp += 1
            flag = ""

        print(f"  {ep:>4} | {tr_loss:>10.4f} | {va_auc:>8.4f} | "
              f"{no_imp:>4} | {time.time()-t0:>8.1f}s | {flag}")

        if no_imp >= PATIENCE:
            print(f"\n  ⏹  Early stopping en época {ep}")
            break

    history["best_auc"]   = best_auc
    history["total_time"] = time.time() - t0_total
    history["epochs_run"] = ep
    print(f"\n  ✔ Mejor AUC val: {best_auc:.4f} | "
          f"Tiempo: {history['total_time']/60:.1f} min")
    return history


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    t_start = time.time()

    print("=" * 68)
    print("  FASE 4 — SOLO_LORA (MLP congelado desde baseline)")
    print(f"  Device: {DEVICE} | LoRA target: FFN fc('0','3') | r={LORA_RANK} | α={LORA_ALPHA}")
    print("=" * 68)

    if PHASE_DONE.exists() and RESULT_JSON.exists():
        print("\n  [SKIP] Fase 4 ya completada.")
        print(f"  Resultado: {RESULT_JSON}")
        exit(0)

    if not CKPT_BASE.exists():
        print(f"\n  ❌ No se encuentra: {CKPT_BASE}")
        print("  Espera a que Fase 2 (BASELINE) termine antes de lanzar Fase 4.")
        exit(1)

    set_seed(SEED)

    # Datos
    assert INDEX_PATH.exists(), f"No se encuentra {INDEX_PATH}"
    df = pd.read_csv(INDEX_PATH)
    df_train = df[df["split"] == "train"].reset_index(drop=True)
    df_val   = df[df["split"] == "val"].reset_index(drop=True)
    print(f"\n  Clips train: {len(df_train):,} | val: {len(df_val):,}")

    kw = dict(num_workers=NUM_WORKERS, pin_memory=True)
    tr_ldr = DataLoader(ClipDataset(df_train), BATCH_VID, shuffle=True,  **kw)
    va_ldr = DataLoader(ClipDataset(df_val),   BATCH_VID, shuffle=False, **kw)

    # Modelos
    print()
    enc, n_lora, n_total = build_encoder_lora()
    mlp = build_mlp_frozen()
    full = FullPipeline(enc, mlp).to(DEVICE)

    # Entrenamiento
    hist = train_solo_lora(full, tr_ldr, va_ldr)
    hist["lora_params"]  = n_lora
    hist["total_params"] = n_total
    hist["lora_pct"]     = round(100 * n_lora / n_total, 3)
    hist["lora_target"]  = "FFN-fc(0,3)"
    hist["lora_rank"]    = LORA_RANK
    hist["lora_alpha"]   = LORA_ALPHA

    # Guardar
    RESULT_JSON.write_text(json.dumps(hist, indent=2))
    PHASE_DONE.touch()

    total = time.time() - t_start
    print(f"\n  ✅ Fase 4 completada.")
    print(f"  Checkpoint : {CKPT_PATH}")
    print(f"  Resultado  : {RESULT_JSON}")
    print(f"  Tiempo total: {total/3600:.2f} h ({total/60:.1f} min)")
