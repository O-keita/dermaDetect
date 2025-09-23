import os
from PIL import Image
import torch
import numpy as np
from transformers import AutoProcessor, AutoModel

device = "cuda" if torch.cuda.is_available() else "cpu"
model = AutoModel.from_pretrained("google/medsiglip-448").to(device)
processor = AutoProcessor.from_pretrained("google/medsiglip-448")

def load_image(path, size=(448, 448)):
    img = Image.open(path).convert("RGB")
    img = img.resize(size)
    return img

def preprocess_image(img):
    return img # not really needed since processor handles it

def extract_embedding(img, model=model, processor=processor, device=device):
    inputs = processor(images=img, text=[""], return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    embedding = outputs.image_embeds.squeeze().cpu().numpy()
    return embedding