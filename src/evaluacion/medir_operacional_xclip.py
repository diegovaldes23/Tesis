"""
medir_operacional_xclip.py
─────────────────────────────────────────────────────────────
Script independiente para medir GFLOPs y métricas operacionales
del pipeline X-CLIP sin re-ejecutar entrenamiento ni evaluación.

Requiere checkpoints en processed/xclip_results/
─────────────────────────────────────────────────────────────
Ejecutar:
    python3 medir_operacional_xclip.py
"""

import json
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from tqdm.auto import tqdm
from transformers import XCLIPModel

# ── Configuración — debe coincidir con xclip_v4.py ───────────────────────────
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
XCLIP_CKPT  = "microsoft/xclip-base-patch32"
OUT_DIR     = Path("processed/xclip_results")
T           = 8
IMG_SIZE    = 224
EMB_DIM     = 512
FRAMES_REAL = 32
STRIDE      = 16
FPS_CAMARA  = 30
UMBRAL_CLIPS_S = FPS_CAMARA / STRIDE    # 1.875 clips/s
N_WARMUP    = 10
N_REPS      = 100

MLP_CFG  = {"hidden_dim": 128, "dropout": 0.3}
LORA_CFG = {"rank": 4, "alpha": 8, "lora_dropout": 0.10,
            "target_mods": ("q_proj", "v_proj")}

print(f"Device : {DEVICE}")
print(f"OUT_DIR: {OUT_DIR}")

# ── Verificar checkpoints ─────────────────────────────────────────────────────
CKPT_BASE = OUT_DIR / "xclip_baseline_best.pth"
CKPT_LORA = OUT_DIR / "xclip_lora_best.pth"
CKPT_SL   = OUT_DIR / "xclip_solo_lora_best.pth"
JSON_PATH = OUT_DIR / "xclip_results.json"

for p in [CKPT_BASE, CKPT_LORA, CKPT_SL, JSON_PATH]:
    assert p.exists(), f"No se encuentra: {p}"
print("Checkpoints verificados ✓")


# ── LoRA (igual que en xclip_v4.py) ──────────────────────────────────────────
class LoRALinear(nn.Module):
    def __init__(self, linear, rank, alpha, lora_dropout=0.0):
        super().__init__()
        device       = linear.weight.device
        self.weight  = linear.weight
        self.bias    = linear.bias
        self.scale   = alpha / rank
        in_f, out_f  = linear.in_features, linear.out_features
        self.lora_A  = nn.Parameter(
            torch.randn(rank, in_f, device=device) * 0.01)
        self.lora_B  = nn.Parameter(
            torch.zeros(out_f, rank, device=device))
        self.dropout = nn.Dropout(p=lora_dropout) \
                       if lora_dropout > 0 else nn.Identity()

    def forward(self, x):
        base = F.linear(x, self.weight, self.bias)
        lora = F.linear(self.dropout(x), self.lora_A)
        return base + F.linear(lora, self.lora_B) * self.scale


def apply_lora(model, rank, alpha, lora_dropout,
               target_mods=("q_proj", "v_proj")):
    to_replace = [
        name for name, mod in model.named_modules()
        if isinstance(mod, nn.Linear)
        and name.split(".")[-1] in target_mods
    ]
    for name in to_replace:
        parts  = name.split(".")
        parent = model
        for part in parts[:-1]:
            parent = getattr(parent, part)
        orig = getattr(parent, parts[-1])
        setattr(parent, parts[-1],
                LoRALinear(orig, rank, alpha, lora_dropout))
    print(f"  LoRA aplicado en {len(to_replace)} capas")
    return model


# ── MLP (igual que en xclip_v4.py) ───────────────────────────────────────────
class AnomalyMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(EMB_DIM, MLP_CFG["hidden_dim"]),
            nn.BatchNorm1d(MLP_CFG["hidden_dim"]),
            nn.ReLU(inplace=True),
            nn.Dropout(p=MLP_CFG["dropout"]),
            nn.Linear(MLP_CFG["hidden_dim"], 1),
        )
    def forward(self, x): return self.net(x)


# ── get_video_embedding (igual que en xclip_v4.py) ───────────────────────────
def get_video_embedding(model, x):
    """
    Pipeline completo: vision_model → visual_projection → MIT → (B, 512)
    x: (B, C, T, H, W)
    """
    B, C, Tf, H, W = x.shape
    flat         = x.permute(0, 2, 1, 3, 4).contiguous().view(B * Tf, C, H, W)
    vision_out   = model.vision_model(pixel_values=flat)
    frame_pooled = vision_out.pooler_output
    frame_embeds = model.visual_projection(frame_pooled)
    cls_features = frame_embeds.view(B, Tf, -1)
    mit_out      = model.mit(cls_features)
    if isinstance(mit_out, (tuple, list)):
        video_embeds = mit_out[1] if len(mit_out) > 1 else mit_out[0]
    else:
        video_embeds = mit_out.pooler_output
    return video_embeds


# ── Pipeline completo como nn.Module (necesario para fvcore) ─────────────────
class FullPipelineXCLIP(nn.Module):
    def __init__(self, model, mlp):
        super().__init__()
        self.model = model
        self.mlp   = mlp

    def forward(self, x):
        emb = get_video_embedding(self.model, x)
        return self.mlp(emb)


# ── Función de medición ───────────────────────────────────────────────────────
def medir(model, mlp, label):
    model.eval(); mlp.eval()
    dummy = torch.randn(1, 3, T, IMG_SIZE, IMG_SIZE, device=DEVICE)

    # Warmup
    with torch.no_grad():
        for _ in range(N_WARMUP):
            get_video_embedding(model, dummy)
            mlp(get_video_embedding(model, dummy))
    if DEVICE.type == "cuda":
        torch.cuda.synchronize()

    # Medición
    times = []
    for _ in tqdm(range(N_REPS), desc=f"  {label}", leave=False):
        t0 = time.perf_counter()
        with torch.no_grad():
            emb = get_video_embedding(model, dummy)
            mlp(emb)
        if DEVICE.type == "cuda":
            torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)

    lat_ms   = float(np.mean(times) * 1000)
    clips_s  = 1000.0 / lat_ms
    fps_real = clips_s * FRAMES_REAL
    lat_frm  = lat_ms / FRAMES_REAL

    # GFLOPs del pipeline completo
    gflops = None
    try:
        from fvcore.nn import FlopCountAnalysis
        pipe  = FullPipelineXCLIP(model, mlp)
        pipe.eval()
        flops = FlopCountAnalysis(pipe, dummy)
        flops.unsupported_ops_warnings(False)
        flops.uncalled_modules_warnings(False)
        gflops = round(flops.total() / 1e9, 2)
        print(f"  GFLOPs (pipeline completo): {gflops}")
    except Exception as e:
        print(f"  GFLOPs no calculados: {e}")

    res = {
        "latencia_ms_clip"  : round(lat_ms,   2),
        "latencia_ms_frame" : round(lat_frm,   3),
        "clips_por_segundo" : round(clips_s,   1),
        "fps_frames_reales" : round(fps_real,  1),
        "frames_internos"   : T,
        "frames_clip_real"  : FRAMES_REAL,
        "umbral_clips_s"    : round(UMBRAL_CLIPS_S, 2),
        "gflops_pipeline"   : gflops,
        # viable_realtime corregido: clips/s (no fps_real)
        "viable_realtime"   : clips_s >= UMBRAL_CLIPS_S,
    }

    print(f"  {label}")
    print(f"    Clips/s          : {clips_s:.1f}  "
          f"(umbral: {UMBRAL_CLIPS_S:.2f})")
    print(f"    FPS reales (×32) : {fps_real:.1f}")
    print(f"    ms/clip          : {lat_ms:.2f}")
    print(f"    ms/frame real    : {lat_frm:.3f}")
    print(f"    GFLOPs           : {gflops}")
    print(f"    Viable           : {'✔' if res['viable_realtime'] else '✗'}")
    return res


# ═════════════════════════════════════════════════════════════════════════════
# BASELINE — encoder congelado (sin LoRA) + MLP
# ═════════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("  BASELINE — encoder frozen + MLP")
print("="*60)

model_base = XCLIPModel.from_pretrained(XCLIP_CKPT).to(DEVICE)
for p in model_base.parameters():
    p.requires_grad = False
model_base.eval()

mlp_base = AnomalyMLP().to(DEVICE)
# Baseline guarda solo el MLP state dict
mlp_base.load_state_dict(torch.load(CKPT_BASE, map_location=DEVICE))

op_base = medir(model_base, mlp_base, "BASELINE")
del model_base, mlp_base
if DEVICE.type == "cuda": torch.cuda.empty_cache()


# ═════════════════════════════════════════════════════════════════════════════
# LORA+MLP — encoder con LoRA + MLP
# ═════════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("  LORA+MLP — encoder con LoRA + MLP")
print("="*60)

model_lora = XCLIPModel.from_pretrained(XCLIP_CKPT).to(DEVICE)
for p in model_lora.parameters():
    p.requires_grad = False
model_lora = apply_lora(model_lora, **{k: LORA_CFG[k] for k in
             ["rank", "alpha", "lora_dropout", "target_mods"]})

mlp_lora = AnomalyMLP().to(DEVICE)

# LoRA+MLP guarda FullPipelineXCLIP: claves "model.*" y "mlp.*"
ckpt = torch.load(CKPT_LORA, map_location=DEVICE)
model_lora.load_state_dict(
    {k.replace("model.", "", 1): v
     for k, v in ckpt.items() if k.startswith("model.")},
    strict=False)
mlp_lora.load_state_dict(
    {k.replace("mlp.", "", 1): v
     for k, v in ckpt.items() if k.startswith("mlp.")})

op_lora = medir(model_lora, mlp_lora, "LORA+MLP")
del model_lora, mlp_lora
if DEVICE.type == "cuda": torch.cuda.empty_cache()


# ═════════════════════════════════════════════════════════════════════════════
# SOLO_LORA — encoder con LoRA + MLP congelado
# ═════════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("  SOLO_LORA — encoder con LoRA + MLP")
print("="*60)

model_sl = XCLIPModel.from_pretrained(XCLIP_CKPT).to(DEVICE)
for p in model_sl.parameters():
    p.requires_grad = False
model_sl = apply_lora(model_sl, **{k: LORA_CFG[k] for k in
           ["rank", "alpha", "lora_dropout", "target_mods"]})

mlp_sl = AnomalyMLP().to(DEVICE)

ckpt = torch.load(CKPT_SL, map_location=DEVICE)
model_sl.load_state_dict(
    {k.replace("model.", "", 1): v
     for k, v in ckpt.items() if k.startswith("model.")},
    strict=False)
mlp_sl.load_state_dict(
    {k.replace("mlp.", "", 1): v
     for k, v in ckpt.items() if k.startswith("mlp.")})

op_solo = medir(model_sl, mlp_sl, "SOLO_LORA")
del model_sl, mlp_sl
if DEVICE.type == "cuda": torch.cuda.empty_cache()


# ── Actualizar JSON existente ─────────────────────────────────────────────────
print("\n  Actualizando JSON...")
with open(JSON_PATH) as f:
    data = json.load(f)

# Agregar operational por configuración (consistente con TimeSformer)
data["BASELINE"]["operational"]  = op_base
data["LORA+MLP"]["operational"]  = op_lora
data["SOLO_LORA"]["operational"] = op_solo

# Mantener también el campo original por compatibilidad
data["operational_metrics"] = op_lora

with open(JSON_PATH, "w") as f:
    json.dump(data, f, indent=2)


# ── Tabla resumen ─────────────────────────────────────────────────────────────
print("\n" + "="*60)
print(f"  {'Config':<12} {'Clips/s':>8} {'FPS×32':>8} "
      f"{'ms/clip':>8} {'GFLOPs':>8}")
print("  " + "─"*50)
for label, op in [("BASELINE",  op_base),
                  ("LORA+MLP",  op_lora),
                  ("SOLO_LORA", op_solo)]:
    g = f"{op['gflops_pipeline']:.2f}" \
        if op.get("gflops_pipeline") else "N/A"
    print(f"  {label:<12} "
          f"{op['clips_por_segundo']:>8.1f} "
          f"{op['fps_frames_reales']:>8.1f} "
          f"{op['latencia_ms_clip']:>8.2f} "
          f"{g:>8}")
print(f"\n  Umbral mínimo: {UMBRAL_CLIPS_S:.2f} clips/s")
print(f"  JSON actualizado: {JSON_PATH}")
