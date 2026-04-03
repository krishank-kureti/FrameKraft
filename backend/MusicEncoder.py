import os
import h5py
import numpy as np
import chromadb
from sentence_transformers import SentenceTransformer

# ===== CONFIGURATION =====
MUSIC_FOLDER = "/Users/krishankkureti/Documents/FrameKraft/backend/data/music/MillionSongSubset"
CHROMA_PATH = "data/chroma_db"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
BATCH_SIZE = 100
PROGRESS_EVERY = 500
# ===== END CONFIGURATION =====


def build_song_text(title, artist_name, genre, terms, weights):
    """
    Build composite text for embedding a single song.

    Args:
        title: Song title string.
        artist_name: Artist name string.
        genre: Genre string from metadata/songs.
        terms: numpy array of artist_terms (dtype |S256).
        weights: numpy array of artist_terms_weight (float64).

    Returns:
        str: Composite document text.
    """
    text = f"Title: {title} | Artist: {artist_name} | Genre: {genre}"

    if terms.size > 0:
        sorted_indices = np.argsort(weights)[::-1]
        top_terms = [terms[i].decode() for i in sorted_indices[:5]]
        text += f" | Tags: {', '.join(top_terms)}"

    return text


def process_song_file(filepath):
    """
    Read one .h5 file and return (song_id, document_text, metadata_dict).

    Args:
        filepath: Path to a single .h5 song file.

    Returns:
        tuple: (song_id, doc_text, metadata_dict) or None if reading fails.
    """
    try:
        with h5py.File(filepath, "r") as f:
            songs_ds = f["metadata/songs"]
            row = songs_ds[0]

            song_id = row["song_id"].decode() if isinstance(row["song_id"], bytes) else row["song_id"]
            title = row["title"].decode() if isinstance(row["title"], bytes) else row["title"]
            artist_name = row["artist_name"].decode() if isinstance(row["artist_name"], bytes) else row["artist_name"]
            genre = row["genre"].decode() if isinstance(row["genre"], bytes) else row["genre"]

            if not title:
                title = "Unknown Title"
            if not artist_name:
                artist_name = "Unknown Artist"
            if not genre:
                genre = "Unknown"

            terms = f["metadata/artist_terms"][:]
            weights = f["metadata/artist_terms_weight"][:]

            doc_text = build_song_text(title, artist_name, genre, terms, weights)

            tags_str = ""
            if terms.size > 0:
                sorted_indices = np.argsort(weights)[::-1]
                top_terms = [terms[i].decode() for i in sorted_indices[:5]]
                tags_str = ", ".join(top_terms)

            metadata = {
                "song_id": song_id,
                "title": title,
                "artist": artist_name,
                "genre": genre,
                "tags": tags_str
            }

            return song_id, doc_text, metadata

    except Exception as e:
        print(f"  ⚠️  Skipping {filepath}: {str(e)}")
        return None


def encode_music_database():
    """
    Walk all .h5 files in MUSIC_FOLDER, embed them, and store in ChromaDB.
    """
    print("🚀 Starting Music Database Encoding")
    print(f"📂 Music folder: {MUSIC_FOLDER}")
    print(f"💾 ChromaDB path: {CHROMA_PATH}")
    print(f"🤖 Embedding model: {EMBEDDING_MODEL}")
    print("="*80)

    # Collect all .h5 files
    h5_files = []
    for root, dirs, files in os.walk(MUSIC_FOLDER):
        for file in files:
            if file.endswith(".h5"):
                h5_files.append(os.path.join(root, file))

    total = len(h5_files)
    print(f"📊 Found {total} .h5 song files")
    print("="*80)

    if total == 0:
        print("❌ No .h5 files found. Check MUSIC_FOLDER path.")
        return

    # Load embedding model
    print(f"🤖 Loading embedding model ({EMBEDDING_MODEL})...")
    model = SentenceTransformer(EMBEDDING_MODEL)
    print("✅ Model loaded")
    print("="*80)

    # Setup ChromaDB
    os.makedirs(CHROMA_PATH, exist_ok=True)
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    collection = client.get_or_create_collection("songs")

    # Process in batches
    processed = 0
    failed = 0

    for i in range(0, total, BATCH_SIZE):
        batch_files = h5_files[i:i + BATCH_SIZE]
        batch_ids = []
        batch_texts = []
        batch_metadata = []

        for filepath in batch_files:
            result = process_song_file(filepath)
            if result is None:
                failed += 1
                continue

            song_id, doc_text, metadata = result
            batch_ids.append(song_id)
            batch_texts.append(doc_text)
            batch_metadata.append(metadata)

        if batch_ids:
            embeddings = model.encode(batch_texts, show_progress_bar=False)
            collection.add(
                ids=batch_ids,
                embeddings=embeddings.tolist(),
                metadatas=batch_metadata
            )
            processed += len(batch_ids)

        if processed % PROGRESS_EVERY == 0 or processed == total:
            print(f"  Processed {processed}/{total} songs... ({failed} skipped)")

    print("="*80)
    print(f"✅ Encoding complete!")
    print(f"   Total processed: {processed}")
    print(f"   Skipped (errors): {failed}")
    print(f"   ChromaDB stored at: {CHROMA_PATH}")
    print("="*80)


if __name__ == "__main__":
    encode_music_database()
