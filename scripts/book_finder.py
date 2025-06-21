import faiss
import numpy as np
import pickle
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt

# --- File paths ---
EMBEDDINGS_FILE = Path("data/embeddings.npy")
INDEX_FILE = Path("data/faiss_index.index")
ID_FILE = Path("data/image_ids.pkl")
IMAGE_DIR = Path("data/images")

# --- Load data ---
print("📦 Loading FAISS index and metadata...")
index = faiss.read_index(str(INDEX_FILE))
embeddings = np.load(EMBEDDINGS_FILE)
with open(ID_FILE, "rb") as f:
    metadata = pickle.load(f)

# --- Search function ---
def search_similar(image_id, k=5):
    print(f"\n🔍 Finding top {k} similar books for image ID {image_id}...")
    query_vector = embeddings[image_id].reshape(1, -1)
    distances, indices = index.search(query_vector, k + 1)  # include self

    # Exclude the query image itself (0 distance)
    similar_ids = indices[0][1:]
    results = [metadata[i] for i in similar_ids]

    for i, item in enumerate(results, start=1):
        print(f"\nResult #{i}")
        print(f"📘 Title: {item['title']}")
        print(f"📂 Genre: {item['genre']}")
        print(f"🖼️ File: {item['filename']}")

    return results

# --- Optional: Display results ---
def show_images(image_id, results):
    fig, axs = plt.subplots(1, len(results) + 1, figsize=(15, 5))

    # Show query image
    query_path = IMAGE_DIR / metadata[image_id]['filename']
    axs[0].imshow(Image.open(query_path))
    axs[0].set_title("Query Image")
    axs[0].axis("off")

    # Show results
    for i, item in enumerate(results, start=1):
        img_path = IMAGE_DIR / item['filename']
        axs[i].imshow(Image.open(img_path))
        axs[i].set_title(f"Match #{i}")
        axs[i].axis("off")

    plt.tight_layout()
    plt.show()

# --- Main driver ---
if __name__ == "__main__":
    image_id = int(input("Enter an image ID (0 to {}): ".format(len(metadata) - 1)))
    results = search_similar(image_id, k=5)
    show_images(image_id, results)
