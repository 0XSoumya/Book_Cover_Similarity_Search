import streamlit as st
import faiss
import numpy as np
import pickle
from PIL import Image
from pathlib import Path

# --- Load data ---
@st.cache_data
def load_data():
    embeddings = np.load("data/embeddings.npy")
    index = faiss.read_index("data/faiss_index.index")
    with open("data/image_ids.pkl", "rb") as f:
        metadata = pickle.load(f)
    return embeddings, index, metadata

embeddings, index, metadata = load_data()
image_dir = Path("data/images")

# --- Streamlit UI ---
st.title("📚 Book Cover Visual Search")
st.markdown("Select a book to find visually similar covers!")

# Dropdown to select query image
titles = [f"{item['id']:03d} — {item['title'][:60]}" for item in metadata]
selected = st.selectbox("Choose a book:", titles)

query_id = int(selected.split(" — ")[0])
query_meta = metadata[query_id]

# Display query image
st.subheader("🔍 Query Image")
query_path = image_dir / query_meta["filename"]
st.image(str(query_path), caption=query_meta["title"], use_container_width=True)

# Similarity search
k = 5
query_vector = embeddings[query_id].reshape(1, -1)
_, indices = index.search(query_vector, k + 1)
similar_ids = [i for i in indices[0] if i != query_id][:k]  # exclude self

# Display results
st.subheader("📖 Top 5 Visually Similar Books")
cols = st.columns(k)

for col, idx in zip(cols, similar_ids):
    item = metadata[idx]
    img_path = image_dir / item["filename"]
    col.image(str(img_path), use_container_width=True, caption=f"{item['title'][:40]} ({item['genre']})")
