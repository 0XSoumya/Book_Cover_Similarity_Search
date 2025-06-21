import streamlit as st
import faiss
import numpy as np
import pickle
from PIL import Image
from pathlib import Path
import random

# --- Title & Config ---
st.set_page_config(page_title="Judging Books by Their Covers", layout="wide")
st.title("📚 Judging Books by Their Covers")

# --- Load Data ---
@st.cache_data
def load_data():
    embeddings = np.load("data/embeddings.npy")
    index = faiss.read_index("data/faiss_index_cosine.index")
    with open("data/image_ids.pkl", "rb") as f:
        metadata = pickle.load(f)
    return embeddings, index, metadata

embeddings, index, metadata = load_data()
image_dir = Path("data/images")



# --- Session State ---
if "page" not in st.session_state:
    st.session_state.page = 0
if "shuffled_ids" not in st.session_state:
    st.session_state.shuffled_ids = random.sample(range(len(metadata)), len(metadata))
if "selected_id" not in st.session_state:
    st.session_state.selected_id = None

# --- UI Logic ---
def show_book_grid():
    st.subheader("🖼 Pick a Book to Get Started")

    books_per_page = 12
    total_books = len(metadata)
    total_pages = (total_books - 1) // books_per_page + 1
    start = st.session_state.page * books_per_page
    end = start + books_per_page
    page_ids = st.session_state.shuffled_ids[start:end]

    cols = st.columns(4)
    for i, idx in enumerate(page_ids):
        item = metadata[idx]
        img_path = image_dir / item["filename"]
        with cols[i % 4]:
            if st.button("", key=f"book_{idx}"):
                st.session_state.selected_id = idx
            st.image(str(img_path), caption=item["title"][:40], use_container_width=True)

    col1, col2, col3 = st.columns([2, 1, 2])
    with col2:
        if st.button("➡️ Next Page"):
            st.session_state.page = (st.session_state.page + 1) % total_pages

def show_results():
    query_id = st.session_state.selected_id
    query_meta = metadata[query_id]
    query_vector = embeddings[query_id].reshape(1, -1)

    # Similarity search
    k = 5
    _, indices = index.search(query_vector, k + 1)
    similar_ids = [i for i in indices[0] if i != query_id][:k]

    st.subheader("🔍 Your Selected Book")
    st.image(str(image_dir / query_meta["filename"]),
             caption=f"{query_meta['title']} ({query_meta['genre']})",
             width=250)

    st.subheader("📖 Top 5 Visually Similar Books")
    cols = st.columns(k)
    for col, idx in zip(cols, similar_ids):
        item = metadata[idx]
        img_path = image_dir / item["filename"]
        col.image(str(img_path), use_container_width=True,
                  caption=f"{item['title'][:40]} ({item['genre']})")

    st.markdown("---")
    if st.button("🔁 Try Another Book"):
        st.session_state.selected_id = None

# --- Render ---
if st.session_state.selected_id is None:
    show_book_grid()
else:
    show_results()
