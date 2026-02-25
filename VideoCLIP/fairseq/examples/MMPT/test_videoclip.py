import torch
from mmpt.models import MMPTModel

def main():
    # 1) Cargar VideoCLIP con la config de HowTo100M
    model, tokenizer, aligner = MMPTModel.from_pretrained(
        "projects/retri/videoclip/how2.yaml"
    )
    model.eval()

    device = torch.device("cpu")
    model.to(device)

    # 2) Construir un video dummy con el formato que usa VideoCLIP:
    #    B, T, FPS, H, W, C
    #    (1 video, 2 segmentos, 30 fps, 224x224, 3 canales RGB)
    video_frames = torch.randn(1, 2, 30, 224, 224, 3, device=device)

    # 3) Texto de ejemplo
    text = "a person is stealing in a store"

    # Tokenizar texto y alinearlo con el esquema propio de MMPT
    tokenized = tokenizer(text, add_special_tokens=False)["input_ids"]
    caps, cmasks = aligner._build_text_seq(tokenized)

    # Añadir dimensión batch
    caps = caps[None, :].to(device)      # (1, L)
    cmasks = cmasks[None, :].to(device)  # (1, L)

    # 4) Forward: obtener embeddings y score video-texto
    with torch.no_grad():
        output = model(video_frames, caps, cmasks, return_score=True)

    print("=== Salida de VideoCLIP ===")
    print("Keys:", list(output.keys()))
    if "pooled_video" in output:
        print("pooled_video shape:", tuple(output["pooled_video"].shape))
    if "pooled_text" in output:
        print("pooled_text shape:", tuple(output["pooled_text"].shape))
    if "score" in output:
        print("score:", output["score"])

if __name__ == "__main__":
    main()
