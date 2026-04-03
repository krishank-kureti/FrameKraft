import chromadb
from sentence_transformers import SentenceTransformer
from groq import Groq
from dotenv import load_dotenv
from MusicSources import get_audio_preview_url
import os
load_dotenv()
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
MODEL = "llama-3.3-70b-versatile"

# ===== CONFIGURATION =====
CHROMA_PATH = "data/chroma_db"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
# ===== END CONFIGURATION =====

_model = None
_chroma_client = None
_chroma_collection = None


def _get_collection():
    """Lazy-load ChromaDB collection."""
    global _chroma_client, _chroma_collection
    if _chroma_collection is None:
        _chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
        _chroma_collection = _chroma_client.get_or_create_collection("songs")
    return _chroma_collection


def _get_embedder():
    """Lazy-load the sentence transformer embedder."""
    global _model
    if _model is None:
        _model = SentenceTransformer(EMBEDDING_MODEL)
    return _model


def get_music_tags(caption, mood, vibe):
    """
    Ask Groq to extract music genre/mood tags from caption + mood + vibe.

    Args:
        caption (str): The BLIP-generated image caption.
        mood (str): The user's chosen caption mood.
        vibe (str): The user's chosen image editing vibe.

    Returns:
        str: A descriptive phrase for embedding (e.g., "Music genre and mood: indie, mellow, cinematic, atmospheric, reflective").
    """
    prompt = (
        f"Given this image context:\n"
        f"  Caption: \"{caption}\"\n"
        f"  Mood: \"{mood}\"\n"
        f"  Edit vibe: \"{vibe}\"\n\n"
        f"Suggest 5 music genre or mood tags that would match this aesthetic. "
        f"Return ONLY a comma-separated list of tags, nothing else. "
        f"No numbering, no explanation, just the tags.\n\n"
        f"Example output: indie, mellow, acoustic, cinematic, introspective"
    )

    response = groq_client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}]
    )
    tags = response.choices[0].message.content.strip()
    return f"Music genre and mood: {tags}"


def retrieve_songs(tags, k=10):
    """
    Query ChromaDB with embedded tags and return top-k similar songs.
    Also fetches Deezer 30-second preview URLs for each song.

    Args:
        tags (str): Comma-separated music tags query string.
        k (int): Number of songs to retrieve.

    Returns:
        list[dict]: List of song metadata dicts with preview_url.
    """
    collection = _get_collection()
    embedder = _get_embedder()

    query_embedding = embedder.encode(tags).tolist()
    results = collection.query(query_embeddings=[query_embedding], n_results=k)

    songs = []
    for i in range(len(results["ids"][0])):
        metadata = results["metadatas"][0][i]
        title = metadata.get("title", "Unknown")
        artist = metadata.get("artist", "Unknown")
        preview_url = get_audio_preview_url(title, artist)

        songs.append({
            "song_id": results["ids"][0][i],
            "title": title,
            "artist": artist,
            "genre": metadata.get("genre", "Unknown"),
            "tags": metadata.get("tags", ""),
            "preview_url": preview_url
        })

    return songs


def recommend_songs(caption, mood, vibe, k=5):
    """
    Full RAG pipeline: extract music tags from caption+mood+vibe,
    query ChromaDB, return top-k song recommendations.

    Args:
        caption (str): The BLIP-generated image caption.
        mood (str): The user's chosen caption mood.
        vibe (str): The user's chosen image editing vibe.
        k (int): Number of songs to return.

    Returns:
        list[dict]: List of top-k song recommendations with title, artist, genre, tags.
    """
    print("\n🎵 Extracting music tags from context...")
    tags = get_music_tags(caption, mood, vibe)
    print(f"   Tags: {tags}")

    print(f"🔍 Searching for similar songs (top {k})...")
    songs = retrieve_songs(tags, k=k)

    return songs


def run_music_rag(image_caption, mood, edit_vibe, k=5):
    """
    Standalone entry point for testing the music RAG independently.

    Args:
        image_caption (str): BLIP-generated caption.
        mood (str): User's chosen mood.
        edit_vibe (str): User's chosen edit vibe.
        k (int): Number of songs to return.
    """
    print("="*80)
    print("🎶 Music RAG — Standalone Test")
    print("="*80)

    songs = recommend_songs(image_caption, mood, edit_vibe, k=k)

    print(f"\n🎶 Top {len(songs)} Song Recommendations:")
    for i, song in enumerate(songs, start=1):
        print(f"\n  {i}. {song['title']}")
        print(f"     Artist: {song['artist']}")
        print(f"     Genre: {song['genre']}")
        if song["tags"]:
            print(f"     Tags: {song['tags']}")

    print("\n" + "="*80)

    return songs
