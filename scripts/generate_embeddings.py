import open_clip
import torch
import pandas as pd
from PIL import Image
from torchvision import transforms
from pathlib import Path
import numpy as np
import pickle
from tqdm import tqdm

# File paths
METADATA_FILE = Path("data/metadata.csv")
IMAGE_DIR = Path("data/images")
EMBEDDING_FILE = Path("data/embeddings.npy")
ID_FILE = Path("data/image_ids.pkl")

# Set device
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Load OpenCLIP model (ViT-B/32)
model, _, preprocess = open_clip.create_model_and_transforms(
    model_name="ViT-B-32", pretrained="openai"
)
model = model.to(device)
model.eval()

# Load metadata
df = pd.read_csv(METADATA_FILE)

# Store embeddings and metadata
embeddings = []
id_map = []

# Image preprocessing fallback (in case open_clip's fails)
fallback_preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.4815, 0.4578, 0.4082],
                         std=[0.2686, 0.2613, 0.2758])
])

# Embed each image
for _, row in tqdm(df.iterrows(), total=len(df), desc="Embedding images"):
    image_path = IMAGE_DIR / row["image_filename"]
    try:
        image = Image.open(image_path).convert("RGB")
        image_tensor = preprocess(image).unsqueeze(0).to(device)
    except Exception as e:
        print(f"❌ Failed to load image {image_path}: {e}")
        continue

    with torch.no_grad():
        embedding = model.encode_image(image_tensor)
        embedding /= embedding.norm(dim=-1, keepdim=True)
        embeddings.append(embedding.cpu().numpy()[0])

    # Store associated metadata
    id_map.append({
        "id": int(row["id"]),
        "filename": row["image_filename"],
        "title": row["title"],
        "genre": row["genre"]
    })

# Save embeddings and metadata
np.save(EMBEDDING_FILE, np.array(embeddings))
with open(ID_FILE, "wb") as f:
    pickle.dump(id_map, f)

print(f"\n✅ Saved {len(embeddings)} embeddings to '{EMBEDDING_FILE}'")
print(f"✅ Saved ID mapping to '{ID_FILE}'")
