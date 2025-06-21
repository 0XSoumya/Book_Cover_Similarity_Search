# 📚 CoverMatch — Judging Books by Their Cover

CoverMatch is a visual similarity search tool that helps users find books with similar covers using ResNet embeddings and FAISS.

🎯 **Try it live**: [Streamlit App](https://your-app-url.streamlit.app)

---

## 🚀 Features

* Visual similarity search powered by deep learning
* ResNet-based embeddings for sharper and more meaningful results
* Scrapes book covers from 12 popular genres using Open Library
* FAISS indexing with cosine similarity for fast queries
* Interactive UI with Streamlit

---

## 🖼️ Preview

![demo](assets/demo.png)

---

## 📦 Project Structure

```
├── app.py                        # Streamlit app  
├── scripts/                      # Scripts for scraping and preprocessing  
│   ├── open_library.py           # Scrapes Open Library covers  
│   ├── ResNet.py                 # Embeds images using ResNet50  
│   └── FAISS_index_ResNet.py     # Builds FAISS cosine index  
├── data/                         # Data folder  
│   ├── images/                   # Book cover images (excluded from repo)  
│   ├── embeddings.npy            # Saved image embeddings  
│   ├── faiss_index_cosine.index  # FAISS index file  
│   └── metadata_openlibrary.json # Book metadata  
├── requirements.txt  
└── README.md  
```

---

## 🧠 How It Works

1. **Data Collection**
   Scrapes 100 books from each of 12 genres via the Open Library API.

2. **Embedding**
   Uses pre-trained ResNet-50 to generate feature vectors for each book cover.

3. **Indexing**
   Embeddings are normalized and indexed with FAISS using cosine similarity.

4. **Search**
   Users select a book via grid, and the app retrieves visually similar covers.

---

## 🧪 Setup & Run Locally

```bash
git clone https://github.com/yourusername/book-cover-similarity  
cd book-cover-similarity  
pip install -r requirements.txt  
streamlit run app.py  
```

️⚠️ Make sure to generate the required data using the scripts in `scripts/`, since large files like images, `.npy`, and FAISS index are ignored by `.gitignore`.

---

## 📚 Dataset

* **1200 books** from Open Library
* **100 books per genre**, across 12 genres
* Covers + metadata stored in `data/`

---

## 👤 Author

**Soumya Sahoo**
[GitHub](https://github.com/0XSoumya) | [LinkedIn](https://www.linkedin.com/in/0xsoumya/)

---

## �� License

```
MIT License  
```
