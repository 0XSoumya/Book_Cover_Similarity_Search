import faiss
import numpy as np
import pickle
from pathlib import Path

# Paths
EMBEDDINGS_FILE = Path("data/embeddings.npy")
ID_FILE = Path("data/image_ids.pkl")
INDEX_FILE = Path("data/faiss_index.index")

def main():
    # Load embeddings
    print("📥 Loading embeddings...")
    embeddings = np.load(EMBEDDINGS_FILE).astype("float32")

    # Build FAISS index (L2 distance on normalized vectors)
    print("⚙️ Building FAISS index...")
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)

    # Save index
    faiss.write_index(index, str(INDEX_FILE))
    print(f"✅ FAISS index saved to '{INDEX_FILE}'")

    # Confirm size
    print(f"📊 Index contains {index.ntotal} vectors.")

if __name__ == "__main__":
    main()
