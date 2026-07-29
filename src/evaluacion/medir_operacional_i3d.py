#!/usr/bin/env python3
"""
medir_operacional_i3d.py  (corregido, con LoRA real)
-------------------------------------------------------------
Mide GFLOPs, latencia (ms/clip) y throughput (clips/s) de I3D, y reporta
CUANTO agrega LoRA respecto al modelo sin adaptar: en parametros y en GFLOPs.

Arregla el problema anterior: ahora LoRA se aplica con el recorrido robusto
(named_modules), igual que en train_i3d_v2_corregido.py, asi que de verdad
reemplaza las Conv3d 1x1x1 (antes daba "entrenables: 0").

Criterio (igual que TimeSformer y X-CLIP):
  - pipeline completo (codificador + MLP), batch=1, clip aleatorio.
  - GFLOPs con fvcore. Latencia con warmup + N_REPS y synchronize().
  - umbral de tiempo real: clips/s >= 30/STRIDE = 1.875  (== 533 ms/clip).

Ejecutar:
    python3 medir_operacional_i3d.py
"""
import sys
import json
import time
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from tqdm.auto import tqdm

DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")
I3D_REPO     = "/home/DIINF/dvaldes/pytorch-i3d"
WEIGHTS_PATH = "/home/DIINF/dvaldes/models/i3d/rgb_imagenet.pt"
OUT_DIR      = Path("processed/i3d_results")
NUM_FRAMES   = 32
IMG_SIZE     = 224
EMB_DIM      = 1024

# Config de LoRA. Ajustala a la que GANE en la calibracion corregida.
LORA_RANK, LORA_ALPHA, LORA_DROPOUT = 4, 8, 0.10
LORA_TARGETS = ["Mixed_5b", "Mixed_5c"]

MLP_HIDDEN, MLP_DROPOUT = 128, 0.3

FRAMES_REAL    = 32
STRIDE         = 16
UMBRAL_CLIPS_S = 30 / STRIDE      # 1.875 clips/s == 533 ms/clip
N_WARMUP       = 10
N_REPS         = 100

JSON_PATH = OUT_DIR / "i3d_operacional_corregido.json"
OUT_DIR.mkdir(parents=True, exist_ok=True)
print(f"Device: {DEVICE}")

sys.path.append(I3D_REPO)
from pytorch_i3d import InceptionI3d


def load_i3d(freeze=True):
    model = InceptionI3d(400, in_channels=3)
    model.load_state_dict(torch.load(WEIGHTS_PATH, map_location=DEVICE))
    if freeze:
        for p in model.parameters():
            p.requires_grad = False
    model.eval()
    return model.to(DEVICE)


class LoRAConv3d1x1(nn.Module):
    def __init__(self, conv, rank, alpha, dropout):
        super().__init__()
        device      = conv.weight.device
        self.conv   = conv
        for p in self.conv.parameters():
            p.requires_grad = False
        self.scale  = alpha / rank
        self.lora_A = nn.Parameter(torch.randn(rank, conv.in_channels,
                                               device=device) * 0.01)
        self.lora_B = nn.Parameter(torch.zeros(conv.out_channels, rank,
                                               device=device))
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
            continue
        for name, child in list(module.named_modules()):
            if isinstance(child, nn.Conv3d) and child.kernel_size == (1, 1, 1):
                parent = module
                parts  = name.split(".")
                for part in parts[:-1]:
                    parent = getattr(parent, part)
                setattr(parent, parts[-1],
                        LoRAConv3d1x1(child, rank, alpha, dropout))
                replaced += 1
    return replaced


def embed(model, x):
    feats = []
    h = model.avg_pool.register_forward_hook(
        lambda m, inp, out: feats.append(out))
    _ = model(x)
    h.remove()
    return feats[0].mean(dim=[2, 3, 4])


class MLP(nn.Module):
    def __init__(self, d=EMB_DIM, h=MLP_HIDDEN, drop=MLP_DROPOUT):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d, h), nn.BatchNorm1d(h),
            nn.ReLU(), nn.Dropout(drop), nn.Linear(h, 1))
    def forward(self, x): return self.net(x)


class Pipeline(nn.Module):
    def __init__(self, enc, mlp):
        super().__init__()
        self.encoder = enc
        self.mlp     = mlp
    def forward(self, x):
        return self.mlp(embed(self.encoder, x))


def contar_entrenables(enc):
    return sum(p.numel() for p in enc.parameters() if p.requires_grad)


def calcular_gflops(enc, mlp):
    from fvcore.nn import FlopCountAnalysis
    dummy = torch.randn(1, 3, NUM_FRAMES, IMG_SIZE, IMG_SIZE).to(DEVICE)
    try:
        pipe = Pipeline(enc, mlp).eval()
        f = FlopCountAnalysis(pipe, dummy)
    except Exception:
        f = FlopCountAnalysis(enc, dummy)
    f.unsupported_ops_warnings(False)
    f.uncalled_modules_warnings(False)
    return round(f.total() / 1e9, 2)


@torch.no_grad()
def medir_latencia(enc, mlp):
    enc.eval(); mlp.eval()
    dummy = torch.randn(1, 3, NUM_FRAMES, IMG_SIZE, IMG_SIZE).to(DEVICE)
    for _ in range(N_WARMUP):
        mlp(embed(enc, dummy))
    if DEVICE.type == "cuda":
        torch.cuda.synchronize()
    times = []
    for _ in tqdm(range(N_REPS), desc="  latencia", leave=False):
        t0 = time.perf_counter()
        mlp(embed(enc, dummy))
        if DEVICE.type == "cuda":
            torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
    lat_ms = float(np.mean(times) * 1000)
    return lat_ms, 1000.0 / lat_ms


def perfil(con_lora):
    enc = load_i3d(freeze=True)
    n_lora = 0
    if con_lora:
        aplicar_lora(enc, LORA_RANK, LORA_ALPHA, LORA_DROPOUT, LORA_TARGETS)
        n_lora = contar_entrenables(enc)
    mlp = MLP().to(DEVICE)
    gflops = calcular_gflops(enc, mlp)
    lat_ms, clips_s = medir_latencia(enc, mlp)
    del enc, mlp
    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()
    return {"gflops": gflops, "lat_ms": round(lat_ms, 2),
            "clips_s": round(clips_s, 2), "lora_params": n_lora}


print("\n" + "=" * 60)
print("  I3D - metricas operacionales (con LoRA real)")
print("=" * 60)

print("\n  Midiendo SIN LoRA (Baseline)...")
base = perfil(con_lora=False)
print(f"    GFLOPs={base['gflops']}  ms/clip={base['lat_ms']}  "
      f"clips/s={base['clips_s']}")

print("\n  Midiendo CON LoRA...")
lora = perfil(con_lora=True)
print(f"    GFLOPs={lora['gflops']}  ms/clip={lora['lat_ms']}  "
      f"clips/s={lora['clips_s']}  | LoRA params={lora['lora_params']:,}")

d_gflops = round(lora["gflops"] - base["gflops"], 3)
pct_gflops = round(100 * d_gflops / base["gflops"], 3) if base["gflops"] else 0.0

print("\n" + "=" * 60)
print("  CUANTO AGREGA LoRA")
print("=" * 60)
print(f"  Parametros entrenables de LoRA : {lora['lora_params']:,}")
print(f"  GFLOPs sin LoRA                : {base['gflops']}")
print(f"  GFLOPs con LoRA                : {lora['gflops']}")
print(f"  Overhead de LoRA               : +{d_gflops} GFLOPs  ({pct_gflops}%)")

filas = [("Baseline", base), ("LoRA+MLP", lora), ("Solo LoRA", lora)]
print("\n" + "=" * 60)
print(f"  {'Config':<12}{'GFLOPs':>9}{'ms/clip':>10}{'clips/s':>10}{'TR':>6}")
print("  " + "-" * 47)
for name, r in filas:
    tr = "si" if r["clips_s"] >= UMBRAL_CLIPS_S else "NO"
    print(f"  {name:<12}{r['gflops']:>9.2f}{r['lat_ms']:>10.2f}"
          f"{r['clips_s']:>10.2f}{tr:>6}")
print(f"\n  Umbral tiempo real: {UMBRAL_CLIPS_S:.3f} clips/s (== 533 ms/clip)")

payload = {
    "lora_config": {"rank": LORA_RANK, "alpha": LORA_ALPHA,
                    "dropout": LORA_DROPOUT},
    "baseline": base,
    "con_lora": lora,
    "overhead_lora": {"gflops": d_gflops, "pct": pct_gflops,
                      "params": lora["lora_params"]},
}
JSON_PATH.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
print(f"\n  Guardado en: {JSON_PATH}")
