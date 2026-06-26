import os
import faiss
import pickle
import numpy as np
from PIL import Image, ImageFile
import torch
import open_clip

# Fix broken images
ImageFile.LOAD_TRUNCATED_IMAGES = True

# Device
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load model
model, _, preprocess = open_clip.create_model_and_transforms(
    "ViT-B-32", pretrained="openai"
)
model = model.to(device)
model.eval()

# Paths
IMAGE_FOLDER = "images"
INDEX_FOLDER = "index"
os.makedirs(INDEX_FOLDER, exist_ok=True)


# 🔥 Embedding
def get_embedding(image_path):
    image = preprocess(Image.open(image_path).convert("RGB")).unsqueeze(0).to(device)

    with torch.no_grad():
        features = model.encode_image(image)

    emb = features.cpu().numpy()
    emb = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-10)

    return emb.squeeze().astype("float32")


# 🔥 Collect
embeddings = []
image_paths = []

for file in os.listdir(IMAGE_FOLDER):
    if file.lower().endswith((".jpg", ".jpeg", ".png")):
        path = os.path.join(IMAGE_FOLDER, file)

        try:
            emb = get_embedding(path)
            embeddings.append(emb)
            image_paths.append(file)
            print("✅ Added:", file)

        except Exception as e:
            print("❌ Skipped:", file, "|", e)


# Safety
if len(embeddings) == 0:
    raise ValueError("No valid images found")

embeddings = np.array(embeddings)

print("📊 INDEX DIM:", embeddings.shape)


# 🔥 FAISS
dim = embeddings.shape[1]
index = faiss.IndexFlatIP(dim)
index.add(embeddings)


# Save
faiss.write_index(index, f"{INDEX_FOLDER}/faiss.index")

with open(f"{INDEX_FOLDER}/image_paths.pkl", "wb") as f:
    pickle.dump(image_paths, f)


print("🎉 Index built successfully!")