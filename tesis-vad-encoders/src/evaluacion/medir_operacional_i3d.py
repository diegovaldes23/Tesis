#!/usr/bin/env python3
"""
Mide FPS, latencia y GFLOPs de I3D para las 3 configuraciones:
  - BASELINE  : I3D congelado + MLP
  - LORA+MLP  : I3D con LoRA (Mixed_5b, Mixed_5c) + MLP
  - SOLO_LORA : igual que LORA+MLP en inferencia

Ejecutar:
    python3 i3d_metricas_operacionales.py
"""
import sys
import time
import json
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path

# ─── Configuración ────────────────────────────────────────────────
DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")
I3D_REPO     = "/home/DIINF/dvaldes/pytorch-i3d"
WEIGHTS_PATH = "/home/DIINF/dvaldes/models/i3d/rgb_imagenet.pt"
RESULTS_DIR  = Path("processed/results")
NUM_FRAMES   = 32
IMG_SIZE     = 224
N_CLIPS      = 100
OUT_PATH     = RESULTS_DIR / "i3d_operational_full.json"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
print(f"Device: {DEVICE}")

# ─── Cargar I3D ───────────────────────────────────────────────────
sys.path.append(I3D_REPO)
from pytorch_i3d import InceptionI3d

def load_i3d(freeze=True):
    model = InceptionI3d(400, in_channels=3)
    state = torch.load(WEIGHTS_PATH, map_location=DEVICE)
    model.load_state_dict(state)
    if freeze:
        for p in model.parameters():
            p.requires_grad = False
    model.eval()
    return model.to(DEVICE)

# ─── Embedding via hook ───────────────────────────────────────────
def get_embedding(model, x):
    feats = []
    h = model.avg_pool.register_forward_hook(
        lambda m, inp, out: feats.append(out))
    with torch.no_grad():
        _ = model(x)
    h.remove()
    return feats[0].mean(dim=[2, 3, 4])  # (B, 1024)

# ─── MLP ──────────────────────────────────────────────────────────
class AnomalyMLP(nn.Module):
    def __init__(self, d=1024, h=128, drop=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d, h), nn.BatchNorm1d(h),
            nn.ReLU(), nn.Dropout(drop), nn.Linear(h, 1))
    def forward(self, x): return self.net(x)

# ─── LoRA sobre I3D ───────────────────────────────────────────────
class LoRAConv3d1x1(nn.Module):
    def __init__(self, conv, rank=4, alpha=8, dropout=0.10):
        super().__init__()
        device       = conv.weight.device
        self.conv    = conv
        c_in         = conv.in_channels
        c_out        = conv.out_channels
        self.scale   = alpha / rank
        self.lora_A  = nn.Parameter(
            torch.randn(rank, c_in, device=device) * 0.01)
        self.lora_B  = nn.Parameter(
            torch.zeros(c_out, rank, device=device))
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        base = self.conv(x)
        B, C, T, H, W = x.shape
        xr   = x.permute(0, 2, 3, 4, 1).reshape(-1, C)
        lora = xr @ self.lora_A.T @ self.lora_B.T
        lora = lora.reshape(B, T, H, W, -1).permute(0, 4, 1, 2, 3)
        return base + lora * self.scale

def apply_lora_i3d(model, rank=4, alpha=8, dropout=0.10):
    for block_name in ["Mixed_5b", "Mixed_5c"]:
        block = getattr(model, block_name)
        for branch_name in dir(block):
            branch = getattr(block, branch_name, None)
            if not isinstance(branch, nn.Sequential):
                continue
            for i, layer in enumerate(branch):
                if (isinstance(layer, nn.Conv3d) and
                        layer.kernel_size == (1, 1, 1)):
                    branch[i] = LoRAConv3d1x1(
                        layer, rank, alpha, dropout)
    n_lora = sum(p.numel() for p in model.parameters()
                 if p.requires_grad)
    print(f"  LoRA aplicado | entrenables: {n_lora:,}")
    return model

# ─── Medir FPS y latencia ─────────────────────────────────────────
def medir(encoder, mlp, label, n=N_CLIPS):
    encoder.eval()
    mlp.eval()
    dummy = torch.randn(1, 3, NUM_FRAMES, IMG_SIZE, IMG_SIZE).to(DEVICE)

    # Warmup
    for _ in range(10):
        emb = get_embedding(encoder, dummy)
        mlp(emb)
    if DEVICE.type == "cuda":
        torch.cuda.synchronize()

    # Medición
    tiempos = []
    for _ in range(n):
        t0 = time.perf_counter()
        emb = get_embedding(encoder, dummy)
        mlp(emb)
        if DEVICE.type == "cuda":
            torch.cuda.synchronize()
        tiempos.append(time.perf_counter() - t0)

    ms_clip  = float(np.mean(tiempos) * 1000)
    ms_frame = ms_clip / NUM_FRAMES
    fps      = NUM_FRAMES / float(np.mean(tiempos))

    print(f"\n  [{label}]")
    print(f"  ms/clip  : {ms_clip:.2f}")
    print(f"  ms/frame : {ms_frame:.2f}")
    print(f"  FPS      : {fps:.1f}")
    print(f"  Viable   : {'✔ Sí' if fps >= 30 else '✗ No'} (≥30 FPS)")

    return {
        "ms_clip":       round(ms_clip,  2),
        "ms_frame":      round(ms_frame, 2),
        "fps":           round(fps,      1),
        "viable":        fps >= 30,
        "frames_clip":   NUM_FRAMES,
        "n_mediciones":  n,
    }

# ─── GFLOPs ───────────────────────────────────────────────────────
def calcular_gflops(encoder):
    try:
        from thop import profile
        dummy = torch.randn(1, 3, NUM_FRAMES, IMG_SIZE, IMG_SIZE).to(DEVICE)
        flops, _ = profile(encoder, inputs=(dummy,), verbose=False)
        gflops = round(flops / 1e9, 1)
        print(f"  GFLOPs: {gflops}")
        return gflops
    except ImportError:
        print("  [INFO] thop no instalado — GFLOPs no calculado")
        print("         Instalar con: pip install thop --break-system-packages")
        return None

# ─── Main ──────────────────────────────────────────────────────────
print("\n" + "=" * 50)
print("  MÉTRICAS OPERACIONALES — I3D")
print("=" * 50)

results = {}

# ── BASELINE ──────────────────────────────────────────────────────
print("\n--- BASELINE (I3D congelado) ---")
enc_base = load_i3d(freeze=True)
mlp_base = AnomalyMLP().to(DEVICE)

# Cargar pesos del MLP si existen
mlp_ckpt = RESULTS_DIR / "baseline_best.pth"
if mlp_ckpt.exists():
    mlp_base.load_state_dict(torch.load(mlp_ckpt, map_location=DEVICE))
    print(f"  MLP cargado desde {mlp_ckpt}")

gflops_base = calcular_gflops(enc_base)
results["BASELINE"] = medir(enc_base, mlp_base, "BASELINE")
results["BASELINE"]["gflops"] = gflops_base
del enc_base, mlp_base
if DEVICE.type == "cuda": torch.cuda.empty_cache()

# ── LORA+MLP ──────────────────────────────────────────────────────
print("\n--- LORA+MLP (I3D con LoRA) ---")
enc_lora = load_i3d(freeze=True)
apply_lora_i3d(enc_lora, rank=4, alpha=8, dropout=0.10)
mlp_lora = AnomalyMLP().to(DEVICE)

mlp_lora_ckpt = RESULTS_DIR / "lora_mlp_best.pth"
if mlp_lora_ckpt.exists():
    # Cargar solo los pesos del MLP del checkpoint combinado
    state = torch.load(mlp_lora_ckpt, map_location=DEVICE)
    mlp_keys = {k.replace("mlp.", ""): v
                for k, v in state.items() if k.startswith("mlp.")}
    if mlp_keys:
        mlp_lora.load_state_dict(mlp_keys)
        print(f"  MLP cargado desde {mlp_lora_ckpt}")

gflops_lora = calcular_gflops(enc_lora)
results["LORA+MLP"] = medir(enc_lora, mlp_lora, "LORA+MLP")
results["LORA+MLP"]["gflops"] = gflops_lora
del enc_lora, mlp_lora
if DEVICE.type == "cuda": torch.cuda.empty_cache()

# ── SOLO_LORA ─────────────────────────────────────────────────────
# En inferencia Solo LoRA = LORA+MLP (mismo encoder, mismo MLP)
# La diferencia está solo en el entrenamiento, no en la inferencia
print("\n--- SOLO_LORA ---")
print("  [INFO] En inferencia Solo LoRA es idéntico a LORA+MLP")
print("         (mismo encoder con LoRA, mismo MLP)")
print("         Las métricas operacionales son las mismas.")
results["SOLO_LORA"] = results["LORA+MLP"].copy()

# ─── Resumen final ────────────────────────────────────────────────
print("\n" + "=" * 50)
print(f"  {'Config':<12} {'FPS':>8} {'ms/frame':>10} {'GFLOPs':>8}")
print("  " + "─" * 42)
for cfg, r in results.items():
    gf = f"{r['gflops']:.1f}" if r.get('gflops') else "N/A"
    print(f"  {cfg:<12} {r['fps']:>8.1f} {r['ms_frame']:>10.2f} {gf:>8}")

# Guardar
with open(OUT_PATH, "w") as f:
    json.dump(results, f, indent=2)
print(f"\n  Guardado en: {OUT_PATH}")
