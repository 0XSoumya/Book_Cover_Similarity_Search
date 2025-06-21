import faiss
import numpy as np
from pathlib import Path

# Paths
EMBEDDINGS_FILE = Path("data/embeddings.npy")
INDEX_FILE = Path("data/faiss_index_resnet_cosine.index")  # New index for ResNet

def main():
    print("📥 Loading embeddings...")
    embeddings = np.load(EMBEDDINGS_FILE).astype("float32")

    print("🧪 Normalizing for cosine similarity...")
    faiss.normalize_L2(embeddings)

    print("⚙️ Building FAISS index (cosine)...")
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)  # IP = cosine sim (after normalization)
    index.add(embeddings)

    faiss.write_index(index, str(INDEX_FILE))
    print(f"✅ Saved FAISS index to '{INDEX_FILE}'")
    print(f"📊 Index contains {index.ntotal} vectors.")

if __name__ == "__main__":
    main()
