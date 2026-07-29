#!/usr/bin/env python3
"""
medir_operacional_swin.py
-------------------------------------------------------------
Mide GFLOPs, latencia (ms/clip) y throughput (clips/s) de Video Swin 3D,
y reporta cuanto agrega LoRA (parametros y GFLOPs), con el mismo criterio
que I3D, TimeSformer y X-CLIP:
  - pipeline completo (codificador + MLP), batch=1, clip aleatorio.
  - GFLOPs con fvcore. Latencia con warmup + N_REPS y synchronize().
  - umbral de tiempo real: clips/s >= 30/STRIDE = 1.875  (== 533 ms/clip).

LoRA en Swin va sobre las capas fc del FFN (MLP) de cada bloque, igual que
en fase4_solo_lora.py (target_mods "0","3" dentro de ".mlp.").

Ejecutar (en una GPU libre):
    python3 medir_operacional_swin.py
"""
import json
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from tqdm.auto import tqdm
from torchvision.models.video import swin3d_b, Swin3D_B_Weights

DEVICE   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OUT_DIR  = Path("processed/realswin_results")
OUT_DIR.mkdir(parents=True, exist_ok=True)
T        = 32
IMG_SIZE = 224
EMB_DIM  = 1024

# Config LoRA (igual que en fase4_solo_lora.py)
LORA_RANK, LORA_ALPHA, LORA_DROPOUT = 4, 8, 0.10
LORA_TARGETS = ("0", "3")     # fc1, fc2 del FFN

MLP_HIDDEN, MLP_DROPOUT = 128, 0.3

FRAMES_REAL    = 32
STRIDE         = 16
UMBRAL_CLIPS_S = 30 / STRIDE      # 1.875 clips/s == 533 ms/clip
N_WARMUP       = 10
N_REPS         = 100

JSON_PATH = OUT_DIR / "swin_operacional_corregido.json"
print(f"Device: {DEVICE}")


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
        self.lora_A = nn.Parameter(torch.randn(rank, linear.in_features,
                                               device=dev) * 0.01)
        self.lora_B = nn.Parameter(torch.zeros(linear.out_features, rank,
                                               device=dev))
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


def load_swin(con_lora):
    weights = Swin3D_B_Weights.KINETICS400_IMAGENET22K_V1
    model   = swin3d_b(weights=weights)
    model.head = nn.Identity()
    for p in model.parameters():
        p.requires_grad = False
    n_lora = 0
    if con_lora:
        for name, module in list(model.named_modules()):
            if not isinstance(module, nn.Linear):
                continue
            leaf = name.split(".")[-1]
            if leaf in LORA_TARGETS and ".mlp." in name:
                parts  = name.split(".")
                parent = model
                for part in parts[:-1]:
                    parent = getattr(parent, part)
                orig = getattr(parent, parts[-1])
                setattr(parent, parts[-1],
                        LoRALinear(orig, LORA_RANK, LORA_ALPHA, LORA_DROPOUT))
        n_lora = sum(p.numel() for p in model.parameters() if p.requires_grad)
    model.eval()
    return model.to(DEVICE), n_lora


class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(EMB_DIM, MLP_HIDDEN), nn.BatchNorm1d(MLP_HIDDEN),
            nn.ReLU(inplace=True), nn.Dropout(MLP_DROPOUT),
            nn.Linear(MLP_HIDDEN, 1))
    def forward(self, x): return self.net(x)


class FullPipeline(nn.Module):
    def __init__(self, enc, mlp):
        super().__init__()
        self.encoder = enc
        self.mlp     = mlp
    def forward(self, x):
        return self.mlp(self.encoder(x))


def calcular_gflops(full):
    from fvcore.nn import FlopCountAnalysis
    dummy = torch.randn(1, 3, T, IMG_SIZE, IMG_SIZE).to(DEVICE)
    f = FlopCountAnalysis(full, dummy)
    f.unsupported_ops_warnings(False)
    f.uncalled_modules_warnings(False)
    return round(f.total() / 1e9, 2)


@torch.no_grad()
def medir_latencia(full):
    full.eval()
    dummy = torch.randn(1, 3, T, IMG_SIZE, IMG_SIZE).to(DEVICE)
    for _ in range(N_WARMUP):
        full(dummy)
    if DEVICE.type == "cuda":
        torch.cuda.synchronize()
    times = []
    for _ in tqdm(range(N_REPS), desc="  latencia", leave=False):
        t0 = time.perf_counter()
        full(dummy)
        if DEVICE.type == "cuda":
            torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
    lat_ms = float(np.mean(times) * 1000)
    return lat_ms, 1000.0 / lat_ms


def perfil(con_lora):
    enc, n_lora = load_swin(con_lora)
    mlp = MLP().to(DEVICE)
    full = FullPipeline(enc, mlp).to(DEVICE).eval()
    gflops = calcular_gflops(full)
    lat_ms, clips_s = medir_latencia(full)
    del enc, mlp, full
    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()
    return {"gflops": gflops, "lat_ms": round(lat_ms, 2),
            "clips_s": round(clips_s, 2), "lora_params": n_lora}


print("\n" + "=" * 60)
print("  Video Swin 3D - metricas operacionales (con LoRA real)")
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
pct = round(100 * d_gflops / base["gflops"], 3) if base["gflops"] else 0.0

print("\n" + "=" * 60)
print("  CUANTO AGREGA LoRA")
print("=" * 60)
print(f"  Parametros entrenables de LoRA : {lora['lora_params']:,}")
print(f"  GFLOPs sin LoRA                : {base['gflops']}")
print(f"  GFLOPs con LoRA                : {lora['gflops']}")
print(f"  Overhead de LoRA               : +{d_gflops} GFLOPs  ({pct}%)")

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
                    "dropout": LORA_DROPOUT, "target": "FFN-fc(0,3)"},
    "baseline": base, "con_lora": lora,
    "overhead_lora": {"gflops": d_gflops, "pct": pct,
                      "params": lora["lora_params"]},
}
JSON_PATH.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
print(f"\n  Guardado en: {JSON_PATH}")
