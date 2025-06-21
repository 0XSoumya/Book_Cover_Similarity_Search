import faiss
import numpy as np
import pickle
from pathlib import Path

# Paths
EMBEDDINGS_FILE = Path("data/embeddings.npy")
ID_FILE = Path("data/image_ids.pkl")
INDEX_FILE = Path("data/faiss_index_cosine.index")  # New index for cosine

def main():
    print("📥 Loading embeddings...")
    embeddings = np.load(EMBEDDINGS_FILE).astype("float32")

    # ✅ Normalize embeddings for cosine similarity
    print("🧪 Normalizing embeddings for cosine similarity...")
    faiss.normalize_L2(embeddings)  # modifies in-place

    # ✅ Use inner product (cosine sim on normalized vectors)
    print("⚙️ Building cosine-similarity FAISS index...")
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

    # Save index
    faiss.write_index(index, str(INDEX_FILE))
    print(f"✅ FAISS cosine index saved to '{INDEX_FILE}'")

    print(f"📊 Index contains {index.ntotal} vectors.")

if __name__ == "__main__":
    main()
