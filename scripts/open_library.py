import asyncio
import aiohttp
import aiofiles
import os
import json
from pathlib import Path

GENRES = [
    "romance", "mystery", "fantasy", "science fiction", "thriller", "historical fiction",
    "young adult", "horror", "biography", "self help", "travel", "poetry"
]

BOOKS_PER_GENRE = 100
OUTPUT_DIR = Path("data/images")
METADATA_FILE = Path("data/metadata_openlibrary.json")

COVER_URL = "https://covers.openlibrary.org/b/id/{}-L.jpg"

HEADERS = {
    "User-Agent": "Mozilla/5.0"
}

async def fetch_cover(session, cover_id, filepath, idx):
    url = COVER_URL.format(cover_id)
    try:
        async with session.get(url, headers=HEADERS) as resp:
            if resp.status == 200:
                async with aiofiles.open(filepath, "wb") as f:
                    await f.write(await resp.read())
                print(f"    ✅ [{idx:03d}] Saved: {filepath}")
            else:
                print(f"    ❌ [{idx:03d}] Failed (status {resp.status}): {url}")
    except Exception as e:
        print(f"    ❌ [{idx:03d}] Error: {e}")

async def scrape_genre(session, genre):
    genre_slug = genre.lower().replace(" ", "_")
    api_url = f"https://openlibrary.org/subjects/{genre_slug}.json?limit={BOOKS_PER_GENRE}"
    print(f"\n🔍 Scraping genre: {genre}...")
    
    try:
        async with session.get(api_url) as resp:
            data = await resp.json()
    except Exception as e:
        print(f"❌ Failed to fetch data for genre '{genre}': {e}")
        return []

    books = []
    genre_dir = OUTPUT_DIR / genre_slug
    genre_dir.mkdir(parents=True, exist_ok=True)

    for idx, work in enumerate(data.get("works", [])):
        if "cover_id" not in work:
            continue

        title = work.get("title", "Unknown Title")
        cover_id = work["cover_id"]
        filename = f"{idx:03d}.jpg"
        filepath = genre_dir / filename

        await fetch_cover(session, cover_id, filepath, idx)

        books.append({
            "title": title,
            "genre": genre,
            "filename": f"{genre_slug}/{filename}"
        })

    print(f"📚 Finished: {len(books)} books saved for genre: {genre}")
    return books

async def main():
    print("🚀 Starting Open Library scraping...")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    all_books = []

    async with aiohttp.ClientSession() as session:
        for genre in GENRES:
            books = await scrape_genre(session, genre)
            all_books.extend(books)

    async with aiofiles.open(METADATA_FILE, "w", encoding="utf-8") as f:
        await f.write(json.dumps(all_books, indent=2))

    print(f"\n✅ Done! Saved metadata for {len(all_books)} books to '{METADATA_FILE}'")

if __name__ == "__main__":
    asyncio.run(main())
