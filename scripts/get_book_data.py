import os
import requests
import csv
from pathlib import Path
from PIL import Image
from io import BytesIO
import time

# List of genres (edit this list for future runs)
GENRES = [
    "self help",
    "fiction",
    "science fiction",
    "romance",
    "biography",
    "history",
    "mystery",
    "fantasy",
    "horror",
    "travel"
]

MAX_RESULTS_PER_GENRE = 50
BASE_IMAGE_DIR = Path("data/images")
METADATA_FILE = Path("data/metadata.csv")


def sanitize_filename(name):
    return name.lower().replace(" ", "_")


def fetch_books(query, max_results=40):
    books = []
    for start in range(0, max_results, 40):
        url = (
            f"https://www.googleapis.com/books/v1/volumes?q={query}"
            f"&startIndex={start}&maxResults=40"
        )
        response = requests.get(url)
        if response.status_code != 200:
            print(f"⚠️ Error fetching '{query}': {response.status_code}")
            continue
        data = response.json()
        items = data.get("items", [])
        for item in items:
            volume = item.get("volumeInfo", {})
            image_links = volume.get("imageLinks", {})
            books.append({
                "title": volume.get("title", "Unknown"),
                "authors": ", ".join(volume.get("authors", [])),
                "categories": ", ".join(volume.get("categories", [])),
                "thumbnail": image_links.get("thumbnail", "")
            })
        time.sleep(1)  # Be polite to the API
    return books


def download_image(url, save_path):
    try:
        r = requests.get(url, timeout=5)
        if r.status_code == 200:
            img = Image.open(BytesIO(r.content)).convert("RGB")
            img = img.resize((224, 224))
            img.save(save_path)
            return True
    except Exception as e:
        print(f"❌ Failed to download {url}: {e}")
    return False


def get_starting_id():
    """Get the next available image ID based on metadata.csv"""
    if not METADATA_FILE.exists():
        return 0
    with open(METADATA_FILE, newline='', encoding='utf-8') as f:
        return sum(1 for _ in csv.reader(f)) - 1  # Exclude header


def main():
    BASE_IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    METADATA_FILE.parent.mkdir(parents=True, exist_ok=True)

    img_id = get_starting_id()
    all_rows = []

    print("🔍 Starting multi-genre scraping...")

    for genre in GENRES:
        genre_folder = BASE_IMAGE_DIR / sanitize_filename(genre)
        genre_folder.mkdir(parents=True, exist_ok=True)

        print(f"\n📚 Fetching books for genre: {genre}")
        books = fetch_books(genre, MAX_RESULTS_PER_GENRE)
        print(f"🔗 Found {len(books)} books in '{genre}'")

        for book in books:
            url = book["thumbnail"]
            if not url:
                continue
            image_filename = f"{img_id:04d}.jpg"
            relative_path = sanitize_filename(genre) + "/" + image_filename
            save_path = genre_folder / image_filename
            if download_image(url, save_path):
                all_rows.append({
                    "id": img_id,
                    "title": book["title"],
                    "authors": book["authors"],
                    "categories": book["categories"],
                    "image_filename": relative_path,
                    "genre": sanitize_filename(genre)
                })
                img_id += 1

    if all_rows:
        mode = 'a' if METADATA_FILE.exists() else 'w'
        write_header = not METADATA_FILE.exists()

        with open(METADATA_FILE, mode=mode, newline='', encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["id", "title", "authors", "categories", "image_filename", "genre"])
            if write_header:
                writer.writeheader()
            writer.writerows(all_rows)

        print(f"\n✅ Metadata appended. Total new entries: {len(all_rows)}")
    else:
        print("\n⚠️ No new images were downloaded.")


if __name__ == "__main__":
    main()
