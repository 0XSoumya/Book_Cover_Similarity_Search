# 📚 Judging Books by Their Covers

A visually- book recommendation system that finds similar book covers using image embeddings and FAISS. Powered by ResNet and Streamlit.

🔗 **Live Demo:** [judgingbooksbycovers.streamlit.app](https://judgingbooksbycovers.streamlit.app/)

---

## 📁 Folder Structure

```bash
book-cover-similarity/
├── app.py                     # Streamlit app
├── requirements.txt          # Python dependencies
├── .gitignore                # Files excluded from Git
├── data/
│   ├── images/               # Book cover images
│   ├── embeddings.npy        # Image embeddings (numpy array)
│   ├── faiss_index_cosine.index  # FAISS index for cosine similarity
│   └── metadata_openlibrary.json # Book metadata
├── scripts/
│   ├── open_library.py       # Open Library scraping script
│   ├── ResNet.py             # Embedding generator using ResNet
│   └── FAISS_index_ResNet.py # Builds FAISS index
```

---

## 🚀 Features

* 🔍 Visual similarity search using book covers
* 🤖 ResNet for feature extraction
* ⚡ Fast retrieval with FAISS
* 📚 Dataset scraped from Open Library (12 genres × 100 books)
* 🖼 Streamlit interface with randomized book grid

---

## 📦 Setup Instructions

1. **Clone the repo:**

   ```bash
   git clone https://github.com/your-username/book-cover-similarity.git
   cd book-cover-similarity
   ```

2. **Install dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

3. **(Optional) Generate embeddings and FAISS index:**

   ```bash
   python scripts/ResNet.py
   python scripts/FAISS_index_ResNet.py
   ```

4. **Run locally:**

   ```bash
   streamlit run app.py
   ```

---

## 🧠 Tech Stack

* Python
* Streamlit
* FAISS
* PyTorch
* Open Library API

---

## ✨ Future Improvements

* CLIP-based embeddings
* Genre filtering and sorting
* Responsive mobile layout
* Deploy with Hugging Face or Docker

---

## 📸 Preview

![demo screenshot](https://your-screenshot-url.com)

---

## 📜 License

MIT License © 2025 Soumya Sahoo
