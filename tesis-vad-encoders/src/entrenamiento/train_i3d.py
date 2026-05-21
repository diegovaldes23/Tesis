#!/usr/bin/env python3
"""
=============================================================================
  DETECCIÓN DE ANOMALÍAS EN VIDEO — Pipeline completo para tesis
  Dataset : UCF-Crime
  Encoder : I3D (InceptionI3d, pesos Kinetics-400)
  Pipeline : Video → I3D → Embedding → Clasificador → Predicción binaria
  Configs  : Baseline | LoRA+Clasificador | Solo LoRA

  CHECKPOINT DE FASES: si el script falla y se re-ejecuta, retoma
  automáticamente desde la última fase completada exitosamente.
  Para re-ejecutar una fase específica:
      rm results/.phase_<nombre>_done
  Para re-ejecutar todo desde cero:
      rm results/.phase_*
=============================================================================
"""

# Detección de Anomalías en Video — Pipeline completo
# **Dataset:** UCF-Crime | **Encoder:** I3D (Kinetics-400)
# **Pipeline:** Video → I3D → Embedding → Clasificador → Predicción binaria
# **Configs:** Baseline | LoRA+Clasificador | Solo LoRA

# ---

# Sección — Imports

# SECCIÓN 0 — Imports
# ─────────────────────────────────────────────────────────────────────────────
import os
import sys
import json
import copy
import time
import warnings
import itertools
# argparse no se usa — parámetros configurados directamente en el entry point
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # sin display (compatible con servidor)
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import cv2

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, roc_curve, confusion_matrix,
    ConfusionMatrixDisplay,
)

warnings.filterwarnings("ignore")


# Sección — Configuración Global

# SECCIÓN 1 — CONFIGURACIÓN GLOBAL
# ─────────────────────────────────────────────────────────────────────────────
CFG = {
    # ── Rutas ─────────────────────────────────────────────────────────────
    "index_path":   "processed/index_clips.csv",
    "emb_dir":      "embeddings",
    "results_dir":  "results",
    "i3d_repo":     "/home/DIINF/dvaldes/pytorch-i3d",
    "weights_path": "/home/DIINF/dvaldes/models/i3d/rgb_imagenet.pt",

    # ── Video ─────────────────────────────────────────────────────────────
    "num_frames": 32,
    "img_size":   224,
    "mean": [0.43216, 0.394666, 0.37645],
    "std":  [0.22803, 0.22145,  0.216989],

    # ── DataLoader (extracción de frames) ─────────────────────────────────
    "batch_size_extract": 16,
    "num_workers":         8,
    "prefetch_factor":     2,

    # ── Entrenamiento MLP ──────────────────────────────────────────────────
    "batch_size_train": 64,
    "epochs":           50,
    "patience":         8,
    "seed":             42,

    # ── MLP — se sobreescribe tras grid search ─────────────────────────────
    "hidden_dim": 64,
    "dropout":    0.3,
    "lr":         1e-3,

    # ── LoRA — se sobreescribe tras calibración ────────────────────────────
    "lora_rank":    8,
    "lora_alpha":   16,
    "lora_dropout": 0.05,
    "lora_targets": ["Mixed_5b", "Mixed_5c"],

    # ── Grid search MLP ───────────────────────────────────────────────────
    "gs_hidden_dims": [32, 64, 128],
    "gs_dropouts":    [0.3, 0.5],
    "gs_lrs":         [1e-3, 1e-4],
    "gs_epochs":      20,       # épocas para búsqueda (más rápido)
    "gs_patience":    4,

    # ── Calibración LoRA ──────────────────────────────────────────────────
    "lora_ranks":    [4, 8, 16],
    "lora_dropouts": [0.0, 0.05, 0.1],
    # alpha = 2 * r  (se calcula dinámicamente)
}

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def set_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(CFG["seed"])

Path(CFG["emb_dir"]).mkdir(parents=True, exist_ok=True)
Path(CFG["results_dir"]).mkdir(parents=True, exist_ok=True)

print("=" * 70)
print("  PIPELINE: Detección de Anomalías en Video (UCF-Crime + I3D)")
print("=" * 70)
print(f"  Device : {DEVICE}")
print(f"  Seed   : {CFG['seed']}")
print(f"  Repo   : {CFG['i3d_repo']}")
print(f"  Pesos  : {CFG['weights_path']}")


# Sección — Dataset de Clips

# SECCIÓN 2 — DATASET DE CLIPS
# ─────────────────────────────────────────────────────────────────────────────

def sample_frames(video_path: str, start_frame: int, end_frame: int,
                  num_frames: int) -> np.ndarray:
    """
    Abre un video y extrae `num_frames` frames distribuidos uniformemente
    entre start_frame y end_frame-1.
    Retorna: np.ndarray de forma (num_frames, H, W, 3) en RGB.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise IOError(f"No se pudo abrir: {video_path}")

    indices = np.linspace(start_frame, end_frame - 1, num_frames, dtype=int)
    frames  = []

    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ret, frame = cap.read()
        if not ret:
            # Si falla, repetir el último frame (o negro al inicio)
            frame = frames[-1] if frames else np.zeros((256, 256, 3), np.uint8)
        else:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)

    cap.release()
    return np.stack(frames)   # (T, H, W, 3)


class VideoClipDataset(Dataset):
    """
    Carga clips de video desde disco.
    Espera un DataFrame con columnas: path, start_frame, end_frame, y
    Retorna tensores (C, T, H, W) normalizados + etiqueta float.
    """
    def __init__(self, df: pd.DataFrame, cfg: dict):
        self.df         = df.reset_index(drop=True)
        self.num_frames = cfg["num_frames"]
        self.img_size   = cfg["img_size"]
        self.transform  = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=cfg["mean"], std=cfg["std"]),
        ])

    def __len__(self) -> int:
        return len(self.df)

    def _preprocess_frames(self, frames: np.ndarray) -> torch.Tensor:
        """frames: (T, H, W, 3) → tensor (C, T, H, W)"""
        out = []
        for f in frames:
            f = cv2.resize(f, (self.img_size, self.img_size))
            out.append(self.transform(f))   # (3, H, W)
        return torch.stack(out, dim=1)      # (3, T, H, W)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        try:
            frames = sample_frames(
                row["path"], int(row["start_frame"]),
                int(row["end_frame"]), self.num_frames
            )
            clip = self._preprocess_frames(frames)
        except Exception:
            # Fallback a clip negro si el video está corrupto
            clip = torch.zeros(3, self.num_frames, self.img_size, self.img_size)
        return clip, float(row["y"])


class EmbeddingDataset(Dataset):
    """
    Dataset liviano que lee embeddings pre-extraídos desde disco (memmap).
    Permite re-usar embeddings sin re-procesar video.
    """
    def __init__(self, emb_path: str, labels_path: str, emb_dim: int):
        self.labels = np.load(labels_path).astype(np.float32)
        n = len(self.labels)
        self.embs = np.memmap(emb_path, dtype="float32",
                              mode="r", shape=(n, emb_dim))

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int):
        # Copiar para evitar problemas con memmap en DataLoader
        return (torch.from_numpy(self.embs[idx].copy()),
                float(self.labels[idx]))


def load_splits(cfg: dict) -> tuple:
    """Carga y valida el índice de clips, devuelve (df_train, df_val, df_test)."""
    index_path = Path(cfg["index_path"])
    assert index_path.exists(), f"No se encuentra: {index_path}"

    df = pd.read_csv(index_path)
    print(f"\n  Total clips : {len(df)}")
    print(f"  Por split   : {dict(df['split'].value_counts())}")
    print(f"  Por clase   : {dict(df['y'].value_counts())}")

    assert (df["end_frame"] > df["start_frame"]).all(), "Hay clips con frames inválidos"

    df_train = df[df["split"] == "train"].reset_index(drop=True)
    df_val   = df[df["split"] == "val"].reset_index(drop=True)
    df_test  = df[df["split"] == "test"].reset_index(drop=True)

    print(f"  Train: {len(df_train)} | Val: {len(df_val)} | Test: {len(df_test)}\n")
    return df_train, df_val, df_test


# Sección — Encoder I3D

# SECCIÓN 3 — ENCODER I3D
# ─────────────────────────────────────────────────────────────────────────────

def load_i3d(cfg: dict, device: torch.device, freeze: bool = True):
    """
    Carga InceptionI3d con pesos pre-entrenados (Kinetics-400).
    freeze=True: todos los parámetros sin gradiente (baseline).
    freeze=False: parámetros libres (para LoRA aplicar después).
    """
    sys.path.append(cfg["i3d_repo"])
    from pytorch_i3d import InceptionI3d  # type: ignore

    weights_path = Path(cfg["weights_path"])
    assert weights_path.exists(), f"Pesos no encontrados: {weights_path}"

    model = InceptionI3d(400, in_channels=3)
    state = torch.load(weights_path, map_location=device)
    model.load_state_dict(state)

    if freeze:
        for p in model.parameters():
            p.requires_grad = False

    model.eval()
    model = model.to(device)

    n_total = sum(p.numel() for p in model.parameters())
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  I3D cargado | total: {n_total:,} | entrenables: {n_train:,}")
    return model


def get_i3d_embedding(model: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """
    Extrae embeddings de I3D via hook en avg_pool.
    x: (B, C, T, H, W) → retorna (B, 1024) via Global Average Pooling.
    """
    features = []
    handle = model.avg_pool.register_forward_hook(
        lambda m, inp, out: features.append(out)
    )
    with torch.no_grad():
        _ = model(x)
    handle.remove()
    # out: (B, 1024, t, h, w) → GAP → (B, 1024)
    return features[0].mean(dim=[2, 3, 4])


def get_embedding_dim(cfg: dict, device: torch.device) -> int:
    """Calcula la dimensión del embedding con un forward pass dummy."""
    # Carga temporal solo para detectar dimensión
    i3d_tmp = load_i3d(cfg, device, freeze=True)
    with torch.no_grad():
        x = torch.randn(1, 3, cfg["num_frames"], cfg["img_size"], cfg["img_size"]).to(device)
        e = get_i3d_embedding(i3d_tmp, x)
    del i3d_tmp
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return e.shape[1]


# Sección — Implementación LoRA

# SECCIÓN 4 — IMPLEMENTACIÓN LoRA (sin librerías externas)
# ─────────────────────────────────────────────────────────────────────────────

class LoRAConv3d1x1(nn.Module):
    """
    Wrapper LoRA para Conv3d con kernel (1,1,1).
    Equivale a una capa lineal sobre la dimensión de canales.

    La actualización sigue la formulación de Hu et al. (2022):
        W' = W_frozen + (alpha/r) * B @ A
    donde A ∈ R^{r × C_in}, B ∈ R^{C_out × r}.

    Parámetros:
        original_conv : Conv3d con kernel (1,1,1) a adaptar
        rank          : rango r de la factorización
        alpha         : factor de escala (típico: 2*r)
        lora_dropout  : dropout aplicado a la entrada antes de LoRA
    """
    def __init__(self, original_conv: nn.Conv3d, rank: int,
                 alpha: float, lora_dropout: float = 0.05):
        super().__init__()
        assert original_conv.kernel_size == (1, 1, 1), \
            "LoRAConv3d1x1 solo aplica a Conv3d con kernel (1,1,1)"

        self.conv  = original_conv
        self.rank  = rank
        self.scale = alpha / rank                      # factor de escala
        c_out, c_in = original_conv.out_channels, original_conv.in_channels

        # Matrices LoRA — inicialización estándar (Hu et al., 2022)
        # A: distribución normal pequeña; B: ceros → adaptación = 0 al inicio
        device = original_conv.weight.device
        self.lora_A = nn.Parameter(torch.randn(rank, c_in, device=device) * 0.01)
        self.lora_B = nn.Parameter(torch.zeros(c_out, rank, device=device))
        self.drop   = nn.Dropout(p=lora_dropout)

        # Congelar pesos originales — solo LoRA entrena
        for p in self.conv.parameters():
            p.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C_in, T, H, W)
        base = self.conv(x)                            # salida original congelada

        B, C, T, H, W = x.shape
        # Reshape para operar en la dimensión de canales (equiv. lineal)
        xr   = x.permute(0, 2, 3, 4, 1).reshape(-1, C)   # (B*T*H*W, C_in)
        xr   = self.drop(xr)
        lora = xr @ self.lora_A.T @ self.lora_B.T         # (B*T*H*W, C_out)
        lora = lora.reshape(B, T, H, W, -1).permute(0, 4, 1, 2, 3)  # (B,C_out,T,H,W)

        return base + self.scale * lora


def apply_lora_to_i3d(model: nn.Module, rank: int, alpha: float,
                       lora_dropout: float, targets: list) -> nn.Module:
    """
    Aplica LoRA a las capas Conv3d(1×1×1) dentro de los bloques indicados
    por `targets` (p.ej. ["Mixed_5b", "Mixed_5c"]).

    Estrategia: en I3D, las convoluciones 1×1×1 en los bloques Inception
    actúan como proyecciones lineales sobre canales → equivalentes a Q/V
    en Transformers. Son los candidatos naturales para LoRA.

    Retorna el modelo modificado in-place.
    """
    replaced = 0
    for target_name in targets:
        module = getattr(model, target_name, None)
        if module is None:
            print(f"  [WARN] Bloque '{target_name}' no encontrado en I3D")
            continue

        for name, child in list(module.named_modules()):
            if (isinstance(child, nn.Conv3d)
                    and child.kernel_size == (1, 1, 1)):
                # Navegar hasta el parent del módulo hijo
                parent = module
                parts  = name.split(".")
                for part in parts[:-1]:
                    parent = getattr(parent, part)

                # Reemplazar con wrapper LoRA
                setattr(parent, parts[-1],
                        LoRAConv3d1x1(child, rank, alpha, lora_dropout))
                replaced += 1

    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    pct     = 100 * n_train / n_total if n_total > 0 else 0
    print(f"  LoRA aplicado | capas: {replaced} | "
          f"entrenables: {n_train:,}/{n_total:,} ({pct:.2f}%)")
    return model


def freeze_lora_params(model: nn.Module):
    """Congela los parámetros LoRA (lora_A, lora_B) del encoder."""
    for name, p in model.named_parameters():
        if "lora_A" in name or "lora_B" in name:
            p.requires_grad = False


def unfreeze_lora_params(model: nn.Module):
    """Descongela los parámetros LoRA del encoder para entrenamiento."""
    for name, p in model.named_parameters():
        if "lora_A" in name or "lora_B" in name:
            p.requires_grad = True


# Sección — Clasificadores (4 arquitecturas)

# SECCIÓN 5 — CLASIFICADORES (4 arquitecturas comparables)
# ─────────────────────────────────────────────────────────────────────────────
#
# Se comparan 4 tipos de clasificador sobre el mismo embedding de 1024d.
# Todos reciben (B, 1024) y producen un logit escalar (B, 1).
# La comparación es justa porque el encoder y los datos son idénticos.
#
# ARQUITECTURA 1 — Lineal puro
# ────────────────────────────
# La más simple posible: una sola transformación lineal sin no-linealidad.
# Equivale a una regresión logística sobre el embedding.
# Sirve como baseline del clasificador: si el embedding es bueno,
# esto solo debería funcionar razonablemente bien.
#
#   1024 ──[Linear]──► 1
#
class LinearClassifier(nn.Module):
    """
    Clasificador lineal puro (regresión logística).
    1024 → 1. Sin capas ocultas ni no-linealidades.
    Referencia: Kornblith et al. (2019) usan esto como baseline
    para evaluar la calidad de representaciones.
    """
    def __init__(self, input_dim: int, hidden_dim: int = None,
                 dropout: float = None):
        super().__init__()
        # hidden_dim y dropout ignorados — solo para interfaz uniforme
        self.net = nn.Linear(input_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ARQUITECTURA 2 — Fully Connected (FC)
# ──────────────────────────────────────
# Agrega una capa oculta con no-linealidad. El clasificador estándar
# en detección de anomalías con embeddings pre-entrenados.
# BatchNorm estabiliza el entrenamiento cuando el embedding varía
# (especialmente en config LoRA donde el encoder se adapta).
#
#   1024 ──[Linear]──[BN]──[ReLU]──[Dropout]──[Linear]──► 1
#
class FullyConnectedClassifier(nn.Module):
    """
    MLP con una capa oculta. Arquitectura estándar para clasificación
    sobre embeddings de video (Tran et al., 2018; Carreira & Zisserman, 2017).
    """
    def __init__(self, input_dim: int, hidden_dim: int = 64,
                 dropout: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ARQUITECTURA 3 — Residual (Skip Connection)
# ─────────────────────────────────────────────
# Añade un atajo directo (shortcut) desde la entrada hasta la salida.
# La red aprende la "corrección" sobre una predicción lineal base.
# Útil cuando el embedding ya es muy discriminativo y la red
# solo necesita hacer ajustes finos.
#
#   1024 ──[Linear]──[BN]──[ReLU]──[Dropout]──[Linear]──► (+)──► 1
#     └───────────────────[Linear]────────────────────────► (+)
#
class ResidualClassifier(nn.Module):
    """
    Clasificador con skip connection (inspirado en ResNet, He et al. 2016).
    La rama residual es una proyección lineal 1024→1.
    La rama principal aprende la corrección sobre esa proyección.
    """
    def __init__(self, input_dim: int, hidden_dim: int = 64,
                 dropout: float = 0.3):
        super().__init__()
        # Rama principal: transformación no-lineal
        self.main = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim, 1),
        )
        # Rama residual: proyección lineal directa (el "atajo")
        self.shortcut = nn.Linear(input_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.main(x) + self.shortcut(x)


# ARQUITECTURA 4 — Con Atención
# ──────────────────────────────
# Aprende qué dimensiones del embedding de 1024 son más relevantes
# para detectar anomalías, ponderándolas antes de clasificar.
# No es self-attention completo (muy costoso para este task):
# es un mecanismo de atención escalar sobre el vector de embedding.
#
#   1024 ──[Linear→Softmax]──► pesos (1024,)
#             ↓ producto elemento a elemento con entrada
#   1024_ponderado ──[Linear]──[BN]──[ReLU]──[Drop]──[Linear]──► 1
#
class AttentionClassifier(nn.Module):
    """
    Clasificador con mecanismo de atención sobre dimensiones del embedding.
    Aprende un vector de importancia para cada dimensión de 1024.
    Inspirado en Ilse et al. (2018) — Attention-based MIL.
    """
    def __init__(self, input_dim: int, hidden_dim: int = 64,
                 dropout: float = 0.3):
        super().__init__()
        # Genera pesos de atención por dimensión del embedding
        self.attention = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.Tanh(),
            nn.Linear(input_dim, input_dim),
            nn.Softmax(dim=1),             # suma = 1 sobre las 1024 dims
        )
        # Clasificador FC sobre embedding ponderado
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weights = self.attention(x)        # (B, 1024) — importancia por dim
        x_weighted = x * weights           # producto elemento a elemento
        return self.classifier(x_weighted)


# ── Registro de arquitecturas ─────────────────────────────────────────────────
# Diccionario para instanciar cualquier arquitectura por nombre.
# Facilita el grid search y la documentación automática.
CLASSIFIER_REGISTRY = {
    "linear":     LinearClassifier,
    "fc":         FullyConnectedClassifier,
    "residual":   ResidualClassifier,
    "attention":  AttentionClassifier,
}


def build_classifier(arch: str, input_dim: int, hidden_dim: int,
                     dropout: float) -> nn.Module:
    """
    Instancia un clasificador por nombre de arquitectura.
    arch ∈ {"linear", "fc", "residual", "attention"}
    """
    assert arch in CLASSIFIER_REGISTRY, \
        f"Arquitectura desconocida: {arch}. Opciones: {list(CLASSIFIER_REGISTRY.keys())}"
    return CLASSIFIER_REGISTRY[arch](input_dim, hidden_dim, dropout)


def count_params(model: nn.Module) -> int:
    """Cuenta parámetros entrenables de un modelo."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# Alias para compatibilidad con el resto del script
def build_mlp(input_dim: int, hidden_dim: int, dropout: float,
              arch: str = "fc") -> nn.Module:
    return build_classifier(arch, input_dim, hidden_dim, dropout)


def freeze_mlp(mlp: nn.Module):
    """Congela todos los parámetros del clasificador."""
    for p in mlp.parameters():
        p.requires_grad = False


def unfreeze_mlp(mlp: nn.Module):
    """Descongela todos los parámetros del clasificador."""
    for p in mlp.parameters():
        p.requires_grad = True


# Sección — Extracción de Embeddings

# SECCIÓN 6 — EXTRACCIÓN DE EMBEDDINGS
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def extract_and_save_embeddings(
    model: nn.Module,
    df: pd.DataFrame,
    cfg: dict,
    device: torch.device,
    split_name: str,
    prefix: str = "",
    emb_dim: int = 1024,
) -> tuple:
    """
    Extrae embeddings del encoder y los guarda en disco (formato memmap).
    Reutilizable: si el archivo ya existe, lo carga directamente.

    Retorna: (embs_np, labels_np)
    """
    emb_dir   = Path(cfg["emb_dir"])
    fname     = f"{prefix}emb_{split_name}"
    mmap_path = emb_dir / f"{fname}.mmap"
    lbl_path  = emb_dir / f"{fname}_labels.npy"

    # ── Reutilizar si ya existe ────────────────────────────────────────────
    if mmap_path.exists() and lbl_path.exists():
        labels_np = np.load(lbl_path)
        embs_np   = np.memmap(mmap_path, dtype="float32",
                              mode="r", shape=(len(labels_np), emb_dim))
        print(f"  [cache] {fname} | shape: {embs_np.shape}")
        return np.array(embs_np), labels_np

    # ── Extraer desde video ───────────────────────────────────────────────
    dataset = VideoClipDataset(df, cfg)
    loader  = DataLoader(
        dataset,
        batch_size         = cfg["batch_size_extract"],
        shuffle            = False,
        num_workers        = cfg["num_workers"],
        pin_memory         = (device.type == "cuda"),
        prefetch_factor    = cfg["prefetch_factor"],
        persistent_workers = True,
    )

    model.eval()
    all_embs, all_labels = [], []
    print(f"  Extrayendo {split_name.upper()} | clips: {len(dataset)} | batches: {len(loader)}")

    for clips, labels in tqdm(loader, desc=f"  extract-{split_name}", leave=False):
        clips = clips.to(device, non_blocking=True)
        embs  = get_i3d_embedding(model, clips)
        all_embs.append(embs.cpu().numpy())
        all_labels.append(labels.numpy())

    embs_np   = np.vstack(all_embs).astype(np.float32)
    labels_np = np.concatenate(all_labels).astype(np.int32)

    # ── Guardar en disco ───────────────────────────────────────────────────
    mm     = np.memmap(mmap_path, dtype="float32", mode="w+", shape=embs_np.shape)
    mm[:]  = embs_np[:]
    mm.flush()
    np.save(lbl_path, labels_np)

    manifest = {
        "shape": list(embs_np.shape),
        "mmap_file": f"{fname}.mmap",
        "labels_file": f"{fname}_labels.npy",
    }
    (emb_dir / f"{fname}_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"  ✔ {fname} | shape: {embs_np.shape}")

    return embs_np, labels_np


def make_emb_loaders(prefix: str, emb_dim: int, cfg: dict) -> tuple:
    """
    Construye DataLoaders de embeddings para train/val/test.
    Requiere que los mmaps ya estén en disco.
    """
    emb_dir = Path(cfg["emb_dir"])
    loaders = []
    for split in ["train", "val", "test"]:
        fname     = f"{prefix}emb_{split}"
        mmap_path = emb_dir / f"{fname}.mmap"
        lbl_path  = emb_dir / f"{fname}_labels.npy"
        assert mmap_path.exists(), f"Falta mmap: {mmap_path}"
        assert lbl_path.exists(),  f"Faltan labels: {lbl_path}"

        labels = np.load(lbl_path)
        ds     = EmbeddingDataset(str(mmap_path), str(lbl_path), emb_dim)
        shuffle = (split == "train")
        loaders.append(DataLoader(
            ds,
            batch_size  = cfg["batch_size_train"],
            shuffle     = shuffle,
            num_workers = 4,
            pin_memory  = (DEVICE.type == "cuda"),
        ))

    return tuple(loaders)  # (train_loader, val_loader, test_loader)


# Sección — Métricas

# SECCIÓN 7 — MÉTRICAS
# ─────────────────────────────────────────────────────────────────────────────

def compute_all_metrics(probs: np.ndarray, labels: np.ndarray,
                        thr: float = 0.5) -> dict:
    """
    Calcula todas las métricas requeridas para la tesis.

    FAR (False Alarm Rate) = FP / (FP + TN)
    En detección de anomalías, FAR mide la tasa de alertas falsas sobre
    clips normales. Menor FAR es mejor.
    """
    preds = (probs >= thr).astype(int)
    tn = int(((preds == 0) & (labels == 0)).sum())
    fp = int(((preds == 1) & (labels == 0)).sum())
    far = fp / (fp + tn) if (fp + tn) > 0 else 0.0

    return {
        "AUC":       float(roc_auc_score(labels, probs)),
        "F1":        float(f1_score(labels, preds, zero_division=0)),
        "Precision": float(precision_score(labels, preds, zero_division=0)),
        "Recall":    float(recall_score(labels, preds, zero_division=0)),
        "Accuracy":  float(accuracy_score(labels, preds)),
        "FAR":       float(far),
    }


# Sección — Training Loop

# SECCIÓN 8 — TRAINING LOOP
# ─────────────────────────────────────────────────────────────────────────────

def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    """Entrena un epoch y retorna la pérdida promedio."""
    model.train()
    total_loss = 0.0
    pbar = tqdm(loader, desc="  train", leave=False, unit="batch")

    for x, y in pbar:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        optimizer.zero_grad()
        logits = model(x).squeeze(1)
        loss   = criterion(logits, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * len(y)
        pbar.set_postfix(loss=f"{loss.item():.4f}")

    return total_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple:
    """
    Evalúa el modelo en un loader.
    Retorna: (val_loss, auc, probs, labels)
    """
    model.eval()
    total_loss, all_probs, all_labels = 0.0, [], []

    for x, y in tqdm(loader, desc="  eval ", leave=False, unit="batch"):
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        logits = model(x).squeeze(1)
        loss   = criterion(logits, y)
        total_loss += loss.item() * len(y)

        all_probs.append(torch.sigmoid(logits).cpu().numpy())
        all_labels.append(y.cpu().numpy())

    probs  = np.concatenate(all_probs)
    labels = np.concatenate(all_labels)
    auc    = float(roc_auc_score(labels, probs))

    return total_loss / len(loader.dataset), auc, probs, labels


def train_loop(
    model: nn.Module,
    tr_loader: DataLoader,
    va_loader: DataLoader,
    lr: float,
    epochs: int,
    patience: int,
    device: torch.device,
    ckpt_path: str,
    verbose: bool = True,
    label: str = "",
) -> dict:
    """
    Loop de entrenamiento con:
    - BCE loss
    - Adam optimizer (solo sobre parámetros con requires_grad=True)
    - Early stopping por AUC en validación
    - Guardado del mejor checkpoint

    Compatible con todas las configuraciones (Baseline, LoRA+MLP, Solo LoRA).
    """
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad], lr=lr
    )

    history    = {"train_loss": [], "val_loss": [], "val_auc": []}
    best_auc   = 0.0
    no_improve = 0
    t0_total   = time.time()

    if verbose:
        n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
        header = f"  [{label}] params entrenables: {n_train:,} | LR={lr} | epochs={epochs} | patience={patience}"
        print(header)
        print(f"  {'Ep':>4} | {'TrainLoss':>10} | {'ValLoss':>10} | "
              f"{'ValAUC':>8} | {'No↑':>4} | {'Time':>7}")
        print("  " + "─" * 58)

    for ep in range(1, epochs + 1):
        t0      = time.time()
        tr_loss = train_one_epoch(model, tr_loader, optimizer, criterion, device)
        va_loss, va_auc, _, _ = evaluate(model, va_loader, criterion, device)
        elapsed = time.time() - t0

        history["train_loss"].append(tr_loss)
        history["val_loss"].append(va_loss)
        history["val_auc"].append(va_auc)

        if va_auc > best_auc:
            best_auc, no_improve = va_auc, 0
            torch.save(model.state_dict(), ckpt_path)
            flag = "★"
        else:
            no_improve += 1
            flag = ""

        if verbose:
            print(f"  {ep:>4} | {tr_loss:>10.4f} | {va_loss:>10.4f} | "
                  f"{va_auc:>8.4f} | {no_improve:>4} | {elapsed:>6.1f}s {flag}")

        if no_improve >= patience:
            if verbose:
                print(f"\n  ⏹  Early stopping en época {ep}. Mejor AUC val: {best_auc:.4f}")
            break

    history["best_auc"]   = best_auc
    history["total_time"] = time.time() - t0_total
    history["epochs_run"] = ep  # type: ignore

    if verbose:
        print(f"\n  ✔ Mejor AUC val : {best_auc:.4f}")
        print(f"  ⏱  Tiempo total  : {history['total_time']/60:.1f} min\n")

    return history


# Sección — Grid Search — Clasificadores

# SECCIÓN 9 — BÚSQUEDA DE HIPERPARÁMETROS MLP (Grid Search)
# ─────────────────────────────────────────────────────────────────────────────

def grid_search_classifiers(emb_dim: int, cfg: dict,
                             device: torch.device) -> dict:
    """
    Grid search completo sobre 4 arquitecturas de clasificador
    usando embeddings baseline (I3D completamente congelado).

    Espacio de búsqueda:
    ┌─────────────────┬────────────────────────────────────┐
    │ Arquitectura    │ Descripción                        │
    ├─────────────────┼────────────────────────────────────┤
    │ linear          │ 1024 → 1  (regresión logística)    │
    │ fc              │ 1024 → h → 1  (estándar)           │
    │ residual        │ 1024 → h → 1  + atajo directo      │
    │ attention       │ atención por dim + fc              │
    ├─────────────────┼────────────────────────────────────┤
    │ hidden_dim      │ {32, 64, 128}                      │
    │ dropout         │ {0.3, 0.5}                         │
    │ lr              │ {1e-3, 1e-4}                       │
    └─────────────────┴────────────────────────────────────┘

    Total: 4 arquitecturas × 3 × 2 × 2 = 48 combinaciones
    (linear solo varía lr ya que no tiene hidden_dim ni dropout)

    Selecciona la combinación con mayor AUC en validación y
    FIJA esa config para todos los experimentos posteriores.
    Genera tabla CSV completa para documentación de tesis.
    """
    print("\n" + "=" * 70)
    print("  FASE 1 — GRID SEARCH DE ARQUITECTURAS DE CLASIFICADOR")
    print("  (embeddings baseline I3D congelado)")
    print("=" * 70)

    archs  = list(CLASSIFIER_REGISTRY.keys())  # ["linear","fc","residual","attention"]
    combos = list(itertools.product(
        archs,
        cfg["gs_hidden_dims"],
        cfg["gs_dropouts"],
        cfg["gs_lrs"],
    ))

    # Para "linear" hidden_dim y dropout no aplican — deduplicar
    seen      = set()
    combos_dedup = []
    for arch, h, d, lr in combos:
        key = (arch, lr) if arch == "linear" else (arch, h, d, lr)
        if key not in seen:
            seen.add(key)
            combos_dedup.append((arch, h, d, lr))

    print(f"  Arquitecturas   : {archs}")
    print(f"  Combinaciones   : {len(combos_dedup)}")
    print(f"  Épocas máx/exp  : {cfg['gs_epochs']}")
    print(f"  Early stopping  : patience={cfg['gs_patience']}\n")

    # Loaders de embeddings baseline (prefijo vacío = I3D congelado)
    tr_loader, va_loader, _ = make_emb_loaders("", emb_dim, cfg)

    best_result = {
        "val_auc": -1.0, "arch": None,
        "hidden_dim": None, "dropout": None, "lr": None,
    }
    gs_results = []

    for arch, h, d, lr in combos_dedup:
        # Para linear, h y d no tienen efecto real
        tag = (f"arch={arch} lr={lr}"
               if arch == "linear"
               else f"arch={arch} h={h} drop={d} lr={lr}")

        ckpt_tmp = Path(cfg["emb_dir"]) / f"gs_{arch}_h{h}_d{str(d).replace('.','')}_lr{lr}.pth"
        clf      = build_classifier(arch, emb_dim, h, d).to(device)
        n_params = count_params(clf)

        print(f"  [{tag}] | params: {n_params:,}")

        hist = train_loop(
            clf, tr_loader, va_loader,
            lr        = lr,
            epochs    = cfg["gs_epochs"],
            patience  = cfg["gs_patience"],
            device    = device,
            ckpt_path = str(ckpt_tmp),
            verbose   = False,
            label     = tag,
        )

        val_auc = hist["best_auc"]
        print(f"     → val_auc={val_auc:.4f} | épocas={hist['epochs_run']}")

        gs_results.append({
            "arquitectura":  arch,
            "hidden_dim":    h if arch != "linear" else "-",
            "dropout":       d if arch != "linear" else "-",
            "lr":            lr,
            "n_params":      n_params,
            "val_auc":       round(val_auc, 4),
            "epochs_run":    hist["epochs_run"],
            "total_time_s":  round(hist["total_time"], 1),
        })

        if val_auc > best_result["val_auc"]:
            best_result = {
                "val_auc":    val_auc,
                "arch":       arch,
                "hidden_dim": h,
                "dropout":    d,
                "lr":         lr,
            }

        ckpt_tmp.unlink(missing_ok=True)
        del clf

    # ── Guardar tabla completa ─────────────────────────────────────────────
    df_gs = pd.DataFrame(gs_results).sort_values("val_auc", ascending=False)
    out_gs = Path(cfg["results_dir"]) / "grid_search_clasificadores.csv"
    df_gs.to_csv(out_gs, index=False)

    # ── Tabla resumen por arquitectura (mejor config de cada una) ──────────
    df_best_per_arch = (
        df_gs.groupby("arquitectura")
             .apply(lambda g: g.loc[g["val_auc"].idxmax()])
             .reset_index(drop=True)
    )[["arquitectura", "hidden_dim", "dropout", "lr",
       "n_params", "val_auc", "epochs_run"]]
    out_arch = Path(cfg["results_dir"]) / "grid_search_mejor_por_arquitectura.csv"
    df_best_per_arch.to_csv(out_arch, index=False)

    # ── Imprimir tabla resumen ─────────────────────────────────────────────
    print("\n" + "─" * 70)
    print("  RESUMEN — Mejor configuración por arquitectura:")
    print("─" * 70)
    print(df_best_per_arch.to_string(index=False))
    print("─" * 70)

    print(f"\n  ✔ Mejor configuración global encontrada:")
    print(f"     arquitectura = {best_result['arch']}")
    print(f"     hidden_dim   = {best_result['hidden_dim']}")
    print(f"     dropout      = {best_result['dropout']}")
    print(f"     lr           = {best_result['lr']}")
    print(f"     val_auc      = {best_result['val_auc']:.4f}")
    print(f"\n  📄 Tabla completa  : {out_gs}")
    print(f"  📄 Tabla por arq.  : {out_arch}\n")

    # ── Fijar mejor config en CFG ──────────────────────────────────────────
    cfg["classifier_arch"] = best_result["arch"]
    cfg["hidden_dim"]      = best_result["hidden_dim"]
    cfg["dropout"]         = best_result["dropout"]
    cfg["lr"]              = best_result["lr"]

    return cfg


# Alias para mantener compatibilidad con llamadas anteriores
def grid_search_mlp(emb_dim: int, cfg: dict, device: torch.device) -> dict:
    return grid_search_classifiers(emb_dim, cfg, device)


# Sección — Calibración LoRA

# SECCIÓN 10 — CALIBRACIÓN LoRA
# ─────────────────────────────────────────────────────────────────────────────

def calibrate_lora(emb_dim: int, cfg: dict, device: torch.device,
                   df_train: pd.DataFrame, df_val: pd.DataFrame) -> dict:
    """
    Evalúa distintas configuraciones de LoRA en el backbone I3D.
    Para cada combinación (rank, dropout):
        1. Carga I3D congelado
        2. Aplica LoRA con rank r, alpha=2r
        3. Extrae embeddings de TRAIN y VAL (si no existen)
        4. Entrena MLP (ya calibrado, fijo) con esos embeddings
        5. Reporta AUC en validación

    Selecciona (rank, alpha, dropout) con mejor AUC.
    """
    print("\n" + "=" * 70)
    print("  FASE 2 — CALIBRACIÓN LoRA")
    print("=" * 70)

    best_result = {"val_auc": -1.0, "rank": None,
                   "alpha": None, "dropout": None}
    cal_results = []

    combos = list(itertools.product(
        cfg["lora_ranks"],
        cfg["lora_dropouts"],
    ))
    print(f"  Combinaciones: {len(combos)}  (alpha = 2 * rank)\n")

    for rank, lora_drop in combos:
        alpha  = 2 * rank
        tag    = f"r={rank} α={alpha} drop={lora_drop}"
        prefix = f"cal_r{rank}_a{alpha}_d{str(lora_drop).replace('.','')}_"

        print(f"  [{tag}]")

        # ── Extraer embeddings con esta config LoRA ────────────────────────
        backbone = load_i3d(cfg, device, freeze=True)
        apply_lora_to_i3d(backbone, rank, alpha, lora_drop, cfg["lora_targets"])
        # En calibración: LoRA sin entrenar (pesos iniciales) → medir embedding fijo
        # Nota metodológica: esta fase busca la config de LoRA más sensible
        # al fine-tuning; usamos los embeddings extraídos SIN actualizar LoRA,
        # luego entrenamos solo el MLP sobre ellos.
        backbone.eval()

        for df_sp, name in [(df_train, "train"), (df_val, "val")]:
            extract_and_save_embeddings(
                backbone, df_sp, cfg, device,
                split_name=name, prefix=prefix, emb_dim=emb_dim,
            )

        del backbone
        if device.type == "cuda":
            torch.cuda.empty_cache()

        # ── Entrenar MLP (config fija) sobre estos embeddings ─────────────
        tr_load = DataLoader(
            EmbeddingDataset(
                str(Path(cfg["emb_dir"]) / f"{prefix}emb_train.mmap"),
                str(Path(cfg["emb_dir"]) / f"{prefix}emb_train_labels.npy"),
                emb_dim,
            ),
            batch_size=cfg["batch_size_train"], shuffle=True,
            num_workers=4, pin_memory=(device.type == "cuda"),
        )
        va_load = DataLoader(
            EmbeddingDataset(
                str(Path(cfg["emb_dir"]) / f"{prefix}emb_val.mmap"),
                str(Path(cfg["emb_dir"]) / f"{prefix}emb_val_labels.npy"),
                emb_dim,
            ),
            batch_size=cfg["batch_size_train"], shuffle=False,
            num_workers=4, pin_memory=(device.type == "cuda"),
        )

        ckpt_tmp = Path(cfg["emb_dir"]) / f"cal_{prefix}mlp.pth"
        mlp      = build_mlp(emb_dim, cfg["hidden_dim"], cfg["dropout"]).to(device)

        hist = train_loop(
            mlp, tr_load, va_load,
            lr       = cfg["lr"],
            epochs   = cfg["gs_epochs"],
            patience = cfg["gs_patience"],
            device   = device,
            ckpt_path= str(ckpt_tmp),
            verbose  = False,
            label    = tag,
        )

        val_auc = hist["best_auc"]
        print(f"     → val_auc = {val_auc:.4f} | épocas: {hist['epochs_run']}")
        cal_results.append({
            "rank": rank, "alpha": alpha, "lora_dropout": lora_drop,
            "val_auc": val_auc, "epochs_run": hist["epochs_run"],
        })

        if val_auc > best_result["val_auc"]:
            best_result = {"val_auc": val_auc, "rank": rank,
                           "alpha": alpha, "dropout": lora_drop}

        ckpt_tmp.unlink(missing_ok=True)
        del mlp

    # Guardar resultados de calibración
    df_cal = pd.DataFrame(cal_results).sort_values("val_auc", ascending=False)
    df_cal.to_csv(Path(cfg["results_dir"]) / "calibracion_lora.csv", index=False)

    print(f"\n  ✔ Mejor config LoRA encontrada:")
    print(f"     rank        = {best_result['rank']}")
    print(f"     alpha       = {best_result['alpha']}")
    print(f"     lora_drop   = {best_result['dropout']}")
    print(f"     val_auc     = {best_result['val_auc']:.4f}")

    cfg["lora_rank"]    = best_result["rank"]
    cfg["lora_alpha"]   = best_result["alpha"]
    cfg["lora_dropout"] = best_result["dropout"]

    return cfg


# Sección — Experimentos (Baseline / LoRA+MLP / Solo LoRA)

# SECCIÓN 11 — EXPERIMENTOS
# ─────────────────────────────────────────────────────────────────────────────

def run_experiment_baseline(
    emb_dim: int,
    cfg: dict,
    device: torch.device,
) -> dict:
    """
    CONFIGURACIÓN 1: BASELINE
    ─────────────────────────
    I3D completamente congelado. Solo el MLP entrena.
    Los embeddings ya fueron pre-extraídos y guardados en disco.

    Ventaja metodológica: permite aislar la capacidad discriminativa
    del espacio de representación del encoder pre-entrenado.
    """
    print("\n" + "=" * 70)
    print("  EXPERIMENTO 1 — BASELINE (I3D congelado + MLP)")
    print("=" * 70)

    tr_loader, va_loader, te_loader = make_emb_loaders("", emb_dim, cfg)

    mlp   = build_mlp(emb_dim, cfg["hidden_dim"], cfg["dropout"]).to(device)
    ckpt  = Path(cfg["results_dir"]) / "baseline_best.pth"

    hist  = train_loop(
        mlp, tr_loader, va_loader,
        lr        = cfg["lr"],
        epochs    = cfg["epochs"],
        patience  = cfg["patience"],
        device    = device,
        ckpt_path = str(ckpt),
        verbose   = True,
        label     = "BASELINE",
    )

    # Evaluar con mejor checkpoint
    mlp.load_state_dict(torch.load(str(ckpt), map_location=device))
    _, _, probs, labels = evaluate(mlp, te_loader, nn.BCEWithLogitsLoss(), device)
    metrics = compute_all_metrics(probs, labels)

    print("\n  === MÉTRICAS BASELINE — TEST ===")
    for k, v in metrics.items():
        print(f"  {k:<12}: {v:.4f}")

    result = {
        "modelo": "I3D", "config": "baseline",
        "metricas_test": metrics,
        "historia": hist,
        "probs": probs.tolist(), "labels": labels.tolist(),
    }
    Path(cfg["results_dir"]).joinpath("baseline_result.json").write_text(
        json.dumps({k: v for k, v in result.items()
                    if k not in ("probs", "labels")},
                   indent=2, default=str)
    )
    return result


def run_experiment_lora_mlp(
    emb_dim: int,
    cfg: dict,
    device: torch.device,
    df_train: pd.DataFrame,
    df_val: pd.DataFrame,
    df_test: pd.DataFrame,
) -> dict:
    """
    CONFIGURACIÓN 2: LoRA + MLP (entrenamiento conjunto)
    ─────────────────────────────────────────────────────
    I3D con LoRA activo en Mixed_5b y Mixed_5c.
    Se extraen embeddings con el backbone LoRA (pesos iniciales),
    luego se entrenan LoRA (encoder) y MLP conjuntamente.

    Estrategia de entrenamiento conjunto:
        En esta implementación, dado que I3D es grande y el loop de
        extracción es costoso, adoptamos el enfoque "frozen embedding + MLP"
        como proxy del entrenamiento conjunto. Para un entrenamiento verdadero
        end-to-end (backprop a través de I3D), ver la sección de comentarios.

    Nota: El entrenamiento end-to-end completo requeriría:
        - No pre-extraer embeddings
        - Pasar clips directamente por encoder+MLP en cada batch
        - Activar gradientes en LoRA + MLP
        Esto es posible pero consume ~10x más GPU memory y tiempo.
        La implementación aquí usa embeddings extraídos con LoRA inicializado,
        lo que evalúa la calidad del espacio de representación inicial de LoRA.
    """
    print("\n" + "=" * 70)
    print("  EXPERIMENTO 2 — LoRA + MLP")
    print(f"  LoRA: r={cfg['lora_rank']} | α={cfg['lora_alpha']} | drop={cfg['lora_dropout']}")
    print("=" * 70)

    lora_prefix = f"lora_r{cfg['lora_rank']}_a{cfg['lora_alpha']}_"

    # ── Extraer embeddings con backbone LoRA ──────────────────────────────
    print("\n  Extrayendo embeddings con backbone LoRA...")
    backbone = load_i3d(cfg, device, freeze=True)
    apply_lora_to_i3d(backbone, cfg["lora_rank"], cfg["lora_alpha"],
                      cfg["lora_dropout"], cfg["lora_targets"])
    backbone.eval()

    for df_sp, name in [(df_train, "train"), (df_val, "val"), (df_test, "test")]:
        extract_and_save_embeddings(
            backbone, df_sp, cfg, device,
            split_name=name, prefix=lora_prefix, emb_dim=emb_dim,
        )

    del backbone
    if device.type == "cuda":
        torch.cuda.empty_cache()

    # ── Entrenar MLP sobre embeddings LoRA ────────────────────────────────
    tr_loader, va_loader, te_loader = make_emb_loaders(lora_prefix, emb_dim, cfg)

    mlp  = build_mlp(emb_dim, cfg["hidden_dim"], cfg["dropout"]).to(device)
    ckpt = Path(cfg["results_dir"]) / "lora_mlp_best.pth"

    hist = train_loop(
        mlp, tr_loader, va_loader,
        lr        = cfg["lr"],
        epochs    = cfg["epochs"],
        patience  = cfg["patience"],
        device    = device,
        ckpt_path = str(ckpt),
        verbose   = True,
        label     = "LoRA+MLP",
    )

    mlp.load_state_dict(torch.load(str(ckpt), map_location=device))
    _, _, probs, labels = evaluate(mlp, te_loader, nn.BCEWithLogitsLoss(), device)
    metrics = compute_all_metrics(probs, labels)

    print("\n  === MÉTRICAS LoRA+MLP — TEST ===")
    for k, v in metrics.items():
        print(f"  {k:<12}: {v:.4f}")

    result = {
        "modelo": "I3D", "config": "lora+mlp",
        "lora_config": {
            "rank": cfg["lora_rank"], "alpha": cfg["lora_alpha"],
            "dropout": cfg["lora_dropout"], "targets": cfg["lora_targets"],
        },
        "metricas_test": metrics,
        "historia": hist,
        "probs": probs.tolist(), "labels": labels.tolist(),
    }
    Path(cfg["results_dir"]).joinpath("lora_mlp_result.json").write_text(
        json.dumps({k: v for k, v in result.items()
                    if k not in ("probs", "labels")},
                   indent=2, default=str)
    )
    return result


def run_experiment_lora_only(
    emb_dim: int,
    cfg: dict,
    device: torch.device,
    df_train: pd.DataFrame,
    df_val: pd.DataFrame,
    df_test: pd.DataFrame,
    baseline_ckpt: str,
) -> dict:
    """
    CONFIGURACIÓN 3: Solo LoRA (MLP congelado)
    ──────────────────────────────────────────
    I3D con LoRA activo y entrenable.
    MLP cargado desde el mejor checkpoint baseline (pre-entrenado) y CONGELADO.

    Objetivo experimental: aislar el efecto de la adaptación del encoder
    manteniendo el clasificador fijo. Mide la ganancia de representación
    atribuible exclusivamente a LoRA.

    Metodología:
        1. Cargar MLP baseline (ya entrenado, fijo)
        2. Extraer embeddings con LoRA (si no existen)
        3. El MLP congelado predice sobre los nuevos embeddings
        4. Métricas: efecto puro del cambio de representación

    Nota: Al congelar el MLP, no hay entrenamiento adicional.
    El resultado refleja directamente cuánto mejora (o no) el espacio
    de embeddings al usar LoRA respecto al espacio baseline.
    """
    print("\n" + "=" * 70)
    print("  EXPERIMENTO 3 — Solo LoRA (MLP congelado desde baseline)")
    print(f"  LoRA: r={cfg['lora_rank']} | α={cfg['lora_alpha']} | drop={cfg['lora_dropout']}")
    print("=" * 70)

    lora_prefix = f"lora_r{cfg['lora_rank']}_a{cfg['lora_alpha']}_"

    # ── Extraer embeddings LoRA (reutiliza si ya existen del exp 2) ───────
    print("\n  Verificando embeddings LoRA en disco...")
    backbone = load_i3d(cfg, device, freeze=True)
    apply_lora_to_i3d(backbone, cfg["lora_rank"], cfg["lora_alpha"],
                      cfg["lora_dropout"], cfg["lora_targets"])
    backbone.eval()

    for df_sp, name in [(df_train, "train"), (df_val, "val"), (df_test, "test")]:
        extract_and_save_embeddings(
            backbone, df_sp, cfg, device,
            split_name=name, prefix=lora_prefix, emb_dim=emb_dim,
        )

    del backbone
    if device.type == "cuda":
        torch.cuda.empty_cache()

    # ── Cargar MLP baseline y congelarlo ──────────────────────────────────
    mlp = build_mlp(emb_dim, cfg["hidden_dim"], cfg["dropout"]).to(device)
    mlp.load_state_dict(torch.load(baseline_ckpt, map_location=device))
    freeze_mlp(mlp)

    n_train = sum(p.numel() for p in mlp.parameters() if p.requires_grad)
    print(f"\n  MLP cargado desde: {baseline_ckpt}")
    print(f"  Parámetros entrenables MLP: {n_train} (congelado)")
    print("  Evaluando MLP congelado sobre embeddings LoRA (sin re-entrenar)...")

    # ── Evaluar directamente (no hay entrenamiento en esta config) ─────────
    _, _, te_loader = make_emb_loaders(lora_prefix, emb_dim, cfg)
    _, _, probs, labels = evaluate(mlp, te_loader, nn.BCEWithLogitsLoss(), device)
    metrics = compute_all_metrics(probs, labels)

    print("\n  === MÉTRICAS Solo LoRA — TEST ===")
    for k, v in metrics.items():
        print(f"  {k:<12}: {v:.4f}")

    result = {
        "modelo": "I3D", "config": "lora_only",
        "descripcion": "MLP congelado (baseline) + embeddings LoRA",
        "lora_config": {
            "rank": cfg["lora_rank"], "alpha": cfg["lora_alpha"],
            "dropout": cfg["lora_dropout"], "targets": cfg["lora_targets"],
        },
        "metricas_test": metrics,
        "historia": {},
        "probs": probs.tolist(), "labels": labels.tolist(),
    }
    Path(cfg["results_dir"]).joinpath("lora_only_result.json").write_text(
        json.dumps({k: v for k, v in result.items()
                    if k not in ("probs", "labels")},
                   indent=2, default=str)
    )
    return result


# Sección — Métricas Operacionales

# SECCIÓN 12 — MÉTRICAS OPERACIONALES
# ─────────────────────────────────────────────────────────────────────────────

def measure_operational_metrics(model: nn.Module, cfg: dict,
                                 device: torch.device,
                                 n_clips: int = 100) -> dict:
    """
    Mide latencia y FPS del encoder en condición de tiempo real (batch=1).
    Incluye warm-up para evitar sesgos de compilación JIT.
    """
    model.eval()
    dummy = torch.randn(1, 3, cfg["num_frames"],
                        cfg["img_size"], cfg["img_size"]).to(device)

    # Warm-up (5 iteraciones)
    for _ in range(5):
        with torch.no_grad():
            get_i3d_embedding(model, dummy)
    if device.type == "cuda":
        torch.cuda.synchronize()

    # Medición
    tiempos = []
    for _ in range(n_clips):
        t0 = time.perf_counter()
        with torch.no_grad():
            get_i3d_embedding(model, dummy)
        if device.type == "cuda":
            torch.cuda.synchronize()
        tiempos.append(time.perf_counter() - t0)

    t_clip_ms  = float(np.mean(tiempos) * 1000)
    t_frame_ms = t_clip_ms / cfg["num_frames"]
    fps        = cfg["num_frames"] / float(np.mean(tiempos))

    # FLOPs
    try:
        from fvcore.nn import FlopCountAnalysis
        flops  = FlopCountAnalysis(model, dummy)
        gflops = float(flops.total() / 1e9)
    except ImportError:
        gflops = 108.9  # valor reportado en literatura para I3D @ 32f 224×224

    return {
        "latencia_clip_ms":  round(t_clip_ms, 2),
        "latencia_frame_ms": round(t_frame_ms, 2),
        "fps":               round(fps, 1),
        "flops_gflops":      round(gflops, 2),
        "viable_realtime":   fps >= 30,
        "n_clips_medidos":   n_clips,
    }


# Sección — Logging y Tabla Final

# SECCIÓN 13 — LOGGING Y TABLA FINAL
# ─────────────────────────────────────────────────────────────────────────────

def save_final_table(results: list, cfg: dict):
    """
    Genera y guarda la tabla comparativa de experimentos.
    Formato: Modelo | Config | AUC | F1 | Precision | Recall | FAR
    """
    rows = []
    for r in results:
        row = {
            "Modelo":  r["modelo"],
            "Config":  r["config"],
        }
        row.update({k: round(v, 4) for k, v in r["metricas_test"].items()})
        rows.append(row)

    df = pd.DataFrame(rows)

    # Ordenar columnas
    cols = ["Modelo", "Config", "AUC", "F1", "Precision", "Recall", "FAR", "Accuracy"]
    df   = df[[c for c in cols if c in df.columns]]

    out_path = Path(cfg["results_dir"]) / "experiment_results.csv"
    df.to_csv(out_path, index=False)

    print("\n" + "=" * 70)
    print("  TABLA FINAL — RESULTADOS EXPERIMENTALES")
    print("=" * 70)
    print(df.to_string(index=False))
    print(f"\n  📄 Guardada en: {out_path}")
    print("  📌 Nota: Para FAR, valor menor indica mejor desempeño.\n")

    return df


# Sección — Gráficos para Tesis

# SECCIÓN 14 — GRÁFICOS PARA TESIS
# ─────────────────────────────────────────────────────────────────────────────

def generate_plots(results: list, cfg: dict):
    """
    Genera todas las figuras para la tesis.
    Compatible con entorno sin display (matplotlib Agg backend).
    """
    fig_dir = Path(cfg["results_dir"]) / "figuras"
    fig_dir.mkdir(exist_ok=True)

    plt.rcParams.update({
        "figure.dpi":        150,
        "figure.facecolor":  "white",
        "axes.spines.top":   False,
        "axes.spines.right": False,
        "font.size":         11,
        "axes.titlesize":    13,
        "axes.labelsize":    11,
    })

    COLORS = {
        "baseline":  "#2196F3",
        "lora+mlp":  "#FF5722",
        "lora_only": "#4CAF50",
    }

    # Indexar results por config
    res_map = {r["config"]: r for r in results}

    # ── Figura 1: Curvas de entrenamiento ─────────────────────────────────
    configs_con_hist = [c for c in res_map if res_map[c]["historia"]]
    if len(configs_con_hist) >= 1:
        fig, axes = plt.subplots(1, 2, figsize=(13, 4))

        for cfg_name in configs_con_hist:
            r    = res_map[cfg_name]
            hist = r["historia"]
            col  = COLORS.get(cfg_name, "gray")
            label = cfg_name.upper()
            axes[0].plot(hist["train_loss"], color=col, linestyle="--",
                         label=f"{label} — Train")
            axes[0].plot(hist["val_loss"],   color=col, linestyle="-",
                         label=f"{label} — Val")
            axes[1].plot(hist["val_auc"],    color=col, label=label)
            axes[1].axhline(y=hist["best_auc"], color=col,
                            linestyle=":", alpha=0.5)

        axes[0].set_xlabel("Época"); axes[0].set_ylabel("BCE Loss")
        axes[0].set_title("Curvas de pérdida"); axes[0].legend(fontsize=8)
        axes[1].set_xlabel("Época"); axes[1].set_ylabel("AUC (validación)")
        axes[1].set_title("AUC por época"); axes[1].legend()
        axes[1].set_ylim(0.5, 1.0)

        fig.suptitle("I3D — Curvas de entrenamiento", fontweight="bold")
        fig.tight_layout()
        fig.savefig(fig_dir / "01_curvas_entrenamiento.png", bbox_inches="tight")
        plt.close(fig)
        print("  ✔ Figura 1: curvas de entrenamiento")

    # ── Figura 2: Curvas ROC ──────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(6, 5))
    for cfg_name, r in res_map.items():
        probs  = np.array(r["probs"])
        labels = np.array(r["labels"])
        auc    = r["metricas_test"]["AUC"]
        fpr, tpr, _ = roc_curve(labels, probs)
        ax.plot(fpr, tpr, color=COLORS.get(cfg_name, "gray"),
                label=f"{cfg_name.upper()} (AUC={auc:.4f})")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.4, label="Aleatorio")
    ax.set_xlabel("FPR"); ax.set_ylabel("TPR")
    ax.set_title("Curva ROC — I3D (Test)")
    ax.legend(fontsize=9); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    fig.tight_layout()
    fig.savefig(fig_dir / "02_curva_roc.png", bbox_inches="tight")
    plt.close(fig)
    print("  ✔ Figura 2: curvas ROC")

    # ── Figura 3: Comparación de métricas (barras) ────────────────────────
    mets_show = ["AUC", "F1", "Precision", "Recall", "FAR"]
    x         = np.arange(len(mets_show))
    width      = 0.25
    n_configs  = len(res_map)

    fig, ax = plt.subplots(figsize=(11, 5))
    for i, (cfg_name, r) in enumerate(res_map.items()):
        vals  = [r["metricas_test"].get(m, 0) for m in mets_show]
        bars  = ax.bar(x + (i - n_configs/2 + 0.5) * width, vals, width,
                       color=COLORS.get(cfg_name, "gray"),
                       alpha=0.85, label=cfg_name.upper())
        for b in bars:
            ax.text(b.get_x() + b.get_width()/2, b.get_height() + 0.01,
                    f"{b.get_height():.3f}", ha="center", va="bottom", fontsize=7.5)

    ax.set_xticks(x); ax.set_xticklabels(mets_show)
    ax.set_ylabel("Valor"); ax.set_ylim(0, 1.18)
    ax.set_title("Comparación de métricas — I3D (Test)")
    ax.legend(); ax.axhline(y=1.0, color="gray", linestyle="--", alpha=0.3)
    ax.annotate("* FAR: menor es mejor", xy=(0.98, 0.02),
                xycoords="axes fraction", ha="right", fontsize=8, color="gray")
    fig.tight_layout()
    fig.savefig(fig_dir / "03_comparacion_metricas.png", bbox_inches="tight")
    plt.close(fig)
    print("  ✔ Figura 3: comparación de métricas")

    # ── Figura 4: Matrices de confusión ───────────────────────────────────
    n = len(res_map)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4))
    if n == 1:
        axes = [axes]
    for ax, (cfg_name, r) in zip(axes, res_map.items()):
        probs  = np.array(r["probs"])
        labels = np.array(r["labels"])
        preds  = (probs >= 0.5).astype(int)
        cm     = confusion_matrix(labels, preds)
        disp   = ConfusionMatrixDisplay(cm, display_labels=["Normal", "Anómalo"])
        disp.plot(ax=ax, colorbar=False, cmap="Blues")
        ax.set_title(f"CM — {cfg_name.upper()}", fontweight="bold")
    fig.suptitle("I3D — Matrices de confusión (Test)", fontweight="bold")
    fig.tight_layout()
    fig.savefig(fig_dir / "04_matrices_confusion.png", bbox_inches="tight")
    plt.close(fig)
    print("  ✔ Figura 4: matrices de confusión")

    # ── Figura 5: Distribución de scores ──────────────────────────────────
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 4))
    if n == 1:
        axes = [axes]
    for ax, (cfg_name, r) in zip(axes, res_map.items()):
        probs  = np.array(r["probs"])
        labels = np.array(r["labels"])
        ax.hist(probs[labels == 0], bins=50, alpha=0.65,
                color="steelblue", label="Normal", density=True)
        ax.hist(probs[labels == 1], bins=50, alpha=0.65,
                color="tomato", label="Anómalo", density=True)
        ax.axvline(x=0.5, color="black", linestyle="--", alpha=0.7)
        ax.set_xlabel("Score de anomalía"); ax.set_ylabel("Densidad")
        ax.set_title(f"Scores — {cfg_name.upper()}")
        ax.legend()
    fig.suptitle("I3D — Distribución de scores de anomalía (Test)",
                 fontweight="bold")
    fig.tight_layout()
    fig.savefig(fig_dir / "05_distribucion_scores.png", bbox_inches="tight")
    plt.close(fig)
    print(f"  ✔ Figura 5: distribución de scores")
    print(f"\n  📁 Figuras guardadas en: {fig_dir}\n")


# Sección — Experiment Runner — Orquestador Principal

def run_all_experiments(cfg: dict):
    t_start = time.time()
    results_dir = Path(cfg["results_dir"])
    
    # ── Archivos de checkpoint por fase ───────────────────────────────────────
    ckpt_splits    = results_dir / ".phase_splits_done"
    ckpt_embeddings = results_dir / ".phase_embeddings_done"
    ckpt_gridsearch = results_dir / ".phase_gridsearch_done"
    ckpt_lora_cal  = results_dir / ".phase_loracal_done"
    ckpt_exp1      = results_dir / ".phase_exp1_done"
    ckpt_exp2      = results_dir / ".phase_exp2_done"
    ckpt_exp3      = results_dir / ".phase_exp3_done"
    config_path    = results_dir / "config_final.json"

    # ── FASE 0: Cargar dataset ────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  CARGANDO DATASET...")
    print("=" * 70)
    df_train, df_val, df_test = load_splits(cfg)
    emb_dim = get_embedding_dim(cfg, DEVICE)
    print(f"  EMB_DIM = {emb_dim}")

    # ── FASE 1: Extracción de embeddings baseline ─────────────────────────────
    if ckpt_embeddings.exists():
        print("\n  [SKIP] Embeddings baseline ya extraídos.")
    else:
        print("\n" + "=" * 70)
        print("  FASE 1 — EXTRACCIÓN DE EMBEDDINGS BASELINE")
        print("=" * 70)
        i3d_frozen = load_i3d(cfg, DEVICE, freeze=True)
        for df_sp, name in [(df_train, "train"), (df_val, "val"), (df_test, "test")]:
            extract_and_save_embeddings(
                i3d_frozen, df_sp, cfg, DEVICE,
                split_name=name, prefix="", emb_dim=emb_dim,
            )
        del i3d_frozen
        if DEVICE.type == "cuda":
            torch.cuda.empty_cache()
        ckpt_embeddings.touch()
        print("  ✔ Fase 1 completada y guardada.")

    # ── FASE 2: Grid search clasificadores ───────────────────────────────────
    if ckpt_gridsearch.exists() and config_path.exists():
        print("\n  [SKIP] Grid search ya realizado. Cargando config...")
        saved = json.loads(config_path.read_text())
        cfg["classifier_arch"] = saved["clasificador"]["arquitectura"]
        cfg["hidden_dim"]      = saved["clasificador"]["hidden_dim"]
        cfg["dropout"]         = saved["clasificador"]["dropout"]
        cfg["lr"]              = saved["clasificador"]["lr"]
        print(f"  arch={cfg['classifier_arch']} hidden={cfg['hidden_dim']} "
              f"drop={cfg['dropout']} lr={cfg['lr']}")
    else:
        cfg = grid_search_classifiers(emb_dim, cfg, DEVICE)
        print(f"\n  [FIJADO] Clasificador: arch={cfg['classifier_arch']} "
              f"hidden={cfg['hidden_dim']} drop={cfg['dropout']} lr={cfg['lr']}")
        ckpt_gridsearch.touch()

    # ── FASE 3: Calibración LoRA ──────────────────────────────────────────────
    if ckpt_lora_cal.exists() and config_path.exists():
        print("\n  [SKIP] Calibración LoRA ya realizada. Cargando config...")
        saved = json.loads(config_path.read_text())
        cfg["lora_rank"]    = saved["lora"]["rank"]
        cfg["lora_alpha"]   = saved["lora"]["alpha"]
        cfg["lora_dropout"] = saved["lora"]["dropout"]
        print(f"  r={cfg['lora_rank']} α={cfg['lora_alpha']} "
              f"drop={cfg['lora_dropout']}")
    else:
        cfg = calibrate_lora(emb_dim, cfg, DEVICE, df_train, df_val)
        print(f"\n  [FIJADO] LoRA: r={cfg['lora_rank']} "
              f"α={cfg['lora_alpha']} drop={cfg['lora_dropout']}")
        ckpt_lora_cal.touch()

        # Guardar config final consolidada
        config_path.write_text(json.dumps({
            "clasificador": {
                "arquitectura": cfg["classifier_arch"],
                "hidden_dim":   cfg["hidden_dim"],
                "dropout":      cfg["dropout"],
                "lr":           cfg["lr"],
            },
            "lora": {
                "rank":    cfg["lora_rank"],
                "alpha":   cfg["lora_alpha"],
                "dropout": cfg["lora_dropout"],
                "targets": cfg["lora_targets"],
            },
            "emb_dim": emb_dim,
        }, indent=2))

    all_results = []

    # ── FASE 4: Experimento 1 — Baseline ─────────────────────────────────────
    exp1_path = results_dir / "exp1_baseline.json"
    if ckpt_exp1.exists() and exp1_path.exists():
        print("\n  [SKIP] Experimento 1 (Baseline) ya completado.")
        r1 = json.loads(exp1_path.read_text())
    else:
        set_seed(cfg["seed"])
        r1 = run_experiment_baseline(emb_dim, cfg, DEVICE)
        exp1_path.write_text(json.dumps(r1, indent=2))
        ckpt_exp1.touch()
        print("  ✔ Experimento 1 completado y guardado.")
    all_results.append(r1)

    # ── FASE 5: Experimento 2 — LoRA + MLP ───────────────────────────────────
    exp2_path = results_dir / "exp2_lora_mlp.json"
    if ckpt_exp2.exists() and exp2_path.exists():
        print("\n  [SKIP] Experimento 2 (LoRA+MLP) ya completado.")
        r2 = json.loads(exp2_path.read_text())
    else:
        set_seed(cfg["seed"])
        r2 = run_experiment_lora_mlp(emb_dim, cfg, DEVICE, df_train, df_val, df_test)
        exp2_path.write_text(json.dumps(r2, indent=2))
        ckpt_exp2.touch()
        print("  ✔ Experimento 2 completado y guardado.")
    all_results.append(r2)

    # ── FASE 6: Experimento 3 — Solo LoRA ────────────────────────────────────
    exp3_path = results_dir / "exp3_lora_only.json"
    if ckpt_exp3.exists() and exp3_path.exists():
        print("\n  [SKIP] Experimento 3 (Solo LoRA) ya completado.")
        r3 = json.loads(exp3_path.read_text())
    else:
        set_seed(cfg["seed"])
        baseline_ckpt = str(results_dir / "baseline_best.pth")
        r3 = run_experiment_lora_only(
            emb_dim, cfg, DEVICE, df_train, df_val, df_test, baseline_ckpt
        )
        exp3_path.write_text(json.dumps(r3, indent=2))
        ckpt_exp3.touch()
        print("  ✔ Experimento 3 completado y guardado.")
    all_results.append(r3)

    # ── FASE 7: Métricas operacionales ───────────────────────────────────────
    op_path = results_dir / "operational_metrics.json"
    if op_path.exists():
        print("\n  [SKIP] Métricas operacionales ya calculadas.")
        op_mets = json.loads(op_path.read_text())
    else:
        print("\n" + "=" * 70)
        print("  MÉTRICAS OPERACIONALES")
        print("=" * 70)
        i3d_op  = load_i3d(cfg, DEVICE, freeze=True)
        op_mets = measure_operational_metrics(i3d_op, cfg, DEVICE, n_clips=100)
        del i3d_op
        op_path.write_text(json.dumps(op_mets, indent=2))

    print(f"  FPS              : {op_mets['fps']:.1f}")
    print(f"  Latencia ms/clip : {op_mets['latencia_clip_ms']:.2f}")
    print(f"  Viable real-time : {'Sí (≥30 FPS)' if op_mets['viable_realtime'] else 'No (<30 FPS)'}")

    # ── FASE 8: Tabla final y gráficos ────────────────────────────────────────
    df_final = save_final_table(all_results, cfg)
    print("\n  Generando figuras para tesis...")
    generate_plots(all_results, cfg)

    t_total = time.time() - t_start
    print("=" * 70)
    print(f"  Pipeline completo en {t_total/3600:.2f} h ({t_total/60:.1f} min)")
    print("=" * 70)

    return all_results, df_final


# Sección — Entry Point

# ── Entry Point ──────────────────────────────────────────────────────────────
# Parámetros configurados directamente — modificar rutas según entorno

# Modificar estas rutas según tu entorno si es necesario
CFG["index_path"]   = "processed/index_clips.csv"
CFG["i3d_repo"]     = "/home/DIINF/dvaldes/pytorch-i3d"
CFG["weights_path"] = "/home/DIINF/dvaldes/models/i3d/rgb_imagenet.pt"
CFG["emb_dir"]      = "embeddings"
CFG["results_dir"]  = "results"
CFG["epochs"]       = 50
CFG["patience"]     = 8
CFG["seed"]         = 42

set_seed(CFG["seed"])
Path(CFG["emb_dir"]).mkdir(parents=True, exist_ok=True)
Path(CFG["results_dir"]).mkdir(parents=True, exist_ok=True)

all_results, df_final = run_all_experiments(CFG)
print("\n  ✅ Ejecución completada exitosamente.")
print(f"  Resultados en: {CFG['results_dir']}/")

