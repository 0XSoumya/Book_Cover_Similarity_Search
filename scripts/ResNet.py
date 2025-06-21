import os
import json
import torch
import numpy as np
from pathlib import Path
from torchvision import models, transforms
from PIL import Image
from tqdm import tqdm

# Paths
IMAGE_DIR = Path("data/images")
METADATA_FILE = Path("data/metadata_openlibrary.json")
EMBEDDINGS_FILE = Path("data/embeddings.npy")
ID_FILE = Path("data/image_ids.pkl")

# --- Preprocessing ---
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],  # Imagenet means
        std=[0.229, 0.224, 0.225]    # Imagenet stds
    ),
])

# --- Load Pretrained ResNet Model ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
resnet = models.resnet50(pretrained=True)
resnet.fc = torch.nn.Identity()  # Remove the classification layer
resnet = resnet.to(device)
resnet.eval()

# --- Load Metadata ---
with open(METADATA_FILE, "r", encoding="utf-8") as f:
    metadata = json.load(f)

embeddings = []
id_map = []

print(f"🔍 Generating ResNet embeddings for {len(metadata)} images...")

# --- Generate embeddings ---
for idx, entry in enumerate(tqdm(metadata)):
    img_path = IMAGE_DIR / entry["filename"]
    try:
        image = Image.open(img_path).convert("RGB")
        tensor = transform(image).unsqueeze(0).to(device)

        with torch.no_grad():
            embedding = resnet(tensor).squeeze().cpu().numpy()

        embeddings.append(embedding)
        id_map.append({**entry, "id": idx})
    except Exception as e:
        print(f"❌ Failed to process {img_path}: {e}")

# --- Save Embeddings & Metadata ---
np.save(EMBEDDINGS_FILE, np.array(embeddings))
with open(ID_FILE, "wb") as f:
    import pickle
    pickle.dump(id_map, f)

print(f"✅ Saved {len(embeddings)} embeddings to '{EMBEDDINGS_FILE}'")
print(f"✅ Saved ID mapping to '{ID_FILE}'")
