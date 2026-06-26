import faiss
import pickle
import numpy as np
import torch
import open_clip
from PIL import Image

# Device
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load OpenCLIP model
model, _, preprocess = open_clip.create_model_and_transforms(
    "ViT-B-32", pretrained="openai"
)
model = model.to(device)
model.eval()

# Load FAISS index
index = faiss.read_index("index/faiss.index")

if index.ntotal == 0:
    raise ValueError("FAISS index is empty. Run build_index.py first.")

# Load image paths
with open("index/image_paths.pkl", "rb") as f:
    image_paths = pickle.load(f)


# 🔥 IMAGE EMBEDDING
def get_image_embedding(image):
    image = preprocess(image).unsqueeze(0).to(device)

    with torch.no_grad():
        features = model.encode_image(image)

    emb = features.cpu().numpy()
    emb = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-10)

    return emb.squeeze().astype("float32")


# 🔥 TEXT EMBEDDING
def get_text_embedding(text):
    tokens = open_clip.tokenize([text]).to(device)

    with torch.no_grad():
        features = model.encode_text(tokens)

    emb = features.cpu().numpy()
    emb = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-10)

    return emb.squeeze().astype("float32")


# 🔍 IMAGE SEARCH
def search_similar(image, top_k=6):
    emb = get_image_embedding(image).reshape(1, -1)

    scores, indices = index.search(emb, top_k)

    results = []
    for i, score in zip(indices[0], scores[0]):
        if i < len(image_paths):
            results.append({
                "image_path": image_paths[i],
                "score": float(score)
            })

    return results


# 🔍 TEXT SEARCH (FIXED)
def search_by_text(query, top_k=6):
    emb = get_text_embedding(query).reshape(1, -1)

    scores, indices = index.search(emb, top_k)

    results = []
    for i, score in zip(indices[0], scores[0]):
        if i < len(image_paths):
            results.append({
                "image_path": image_paths[i],
                "score": float(score)
            })

    return results