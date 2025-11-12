# Text 
import torch, numpy as np, open_clip
from PIL import Image

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess, tokenizer = open_clip.create_model_and_transforms(
    'ViT-B-32', pretrained='laion2b_s34b_b79k', device=device
)
model.eval()

def embed_text(texts: list[str]) -> np.ndarray:
    with torch.no_grad():
        tok = tokenizer(texts)
        feats = model.encode_text(tok.to(device))
        feats = feats / feats.norm(dim=-1, keepdim=True)
        return feats.cpu().numpy().astype(np.float32)  # e.g., (N, 512)

def embed_image(png_path: str) -> np.ndarray:
    img = preprocess(Image.open(png_path)).unsqueeze(0).to(device)
    with torch.no_grad():
        feats = model.encode_image(img)
        feats = feats / feats.norm(dim=-1, keepdim=True)
        return feats.cpu().numpy().astype(np.float32)  # (1, 512)
