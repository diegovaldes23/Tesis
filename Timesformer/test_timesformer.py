#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
from pathlib import Path
from datetime import datetime

import numpy as np
import torch
from transformers import AutoImageProcessor, TimesformerModel

try:
    import decord
    from decord import VideoReader, cpu
    decord.bridge.set_bridge("torch")
except Exception as e:
    raise RuntimeError(
        "No se pudo importar decord. Instala con: pip install decord\n"
        f"Error: {e}"
    )


def now_iso():
    return datetime.now().isoformat(timespec="seconds")


def sample_frames_uniform(vr: VideoReader, num_frames: int):
    """Muestreo uniforme de frames a lo largo del video."""
    total = len(vr)
    if total <= 0:
        raise ValueError("El video no tiene frames (len(vr)=0).")

    # índices uniformes
    if total < num_frames:
        idxs = np.linspace(0, total - 1, total).astype(int)
    else:
        idxs = np.linspace(0, total - 1, num_frames).astype(int)

    frames = vr.get_batch(idxs)  # torch tensor (T,H,W,3)
    return frames, idxs.tolist(), total


def main():
    parser = argparse.ArgumentParser(description="Validación TimeSformer (video -> embedding).")
    parser.add_argument("--video", required=True, help="Ruta al video (mp4/avi).")
    parser.add_argument(
        "--model",
        default="facebook/timesformer-base-finetuned-k400",
        help="Modelo TimeSformer en Hugging Face."
    )
    parser.add_argument("--num_frames", type=int, default=8, help="Cantidad de frames a muestrear.")
    parser.add_argument("--outdir", default="evidence/timesformer", help="Directorio de evidencia.")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"], help="Dispositivo.")
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # device
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    # metadata base
    meta = {
        "timestamp": now_iso(),
        "model": args.model,
        "video": str(Path(args.video).resolve()),
        "num_frames": args.num_frames,
        "device": device,
        "torch_version": torch.__version__,
    }

    print(f"[TimeSformer] Modelo: {args.model}")
    print(f"[TimeSformer] Video:  {args.video}")
    print(f"[TimeSformer] Device: {device}")

    # load model
    processor = AutoImageProcessor.from_pretrained(args.model)
    model = TimesformerModel.from_pretrained(args.model).to(device)
    model.eval()

    # read video
    vr = VideoReader(args.video, ctx=cpu(0))
    frames_torch, idxs, total_frames = sample_frames_uniform(vr, args.num_frames)

    meta["total_frames"] = int(total_frames)
    meta["sampled_indices"] = idxs

    # Convertir frames a numpy para processor (lista de HxWx3)
    frames_np = frames_torch.cpu().numpy()  # (T,H,W,3)
    inputs = processor(list(frames_np), return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    # outputs.last_hidden_state: [B, tokens, hidden]
    last = outputs.last_hidden_state
    emb = last.mean(dim=1)  # [B, hidden]
    emb_np = emb.detach().cpu().numpy()

    # Guardar embedding
    emb_path = outdir / "embedding_timesformer.npy"
    np.save(emb_path, emb_np)

    meta["embedding_shape"] = list(emb_np.shape)
    meta["embedding_preview"] = emb_np[0, :5].tolist()

    meta_path = outdir / "meta.json"
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")

    print("[TimeSformer] OK")
    print("  - Total frames video:", total_frames)
    print("  - Sampled indices:", idxs)
    print("  - Embedding shape:", emb_np.shape)
    print("  - Guardado:", emb_path)
    print("  - Meta:", meta_path)


if __name__ == "__main__":
    main()
