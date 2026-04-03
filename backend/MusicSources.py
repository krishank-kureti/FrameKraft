import requests


def get_audio_preview_url(title, artist):
    """
    Search iTunes' public API for a song and return its 30-second preview URL.

    Args:
        title (str): Song title.
        artist (str): Artist name.

    Returns:
        str or None: iTunes 30-second preview URL (M4V/MP4 video), or None if not found.
    """
    try:
        url = "https://itunes.apple.com/search"
        params = {
            "term": f"{title} {artist}",
            "media": "music",
            "entity": "song",
            "limit": 1
        }
        response = requests.get(url, params=params, timeout=5)
        data = response.json()

        if data.get("results") and data["results"][0].get("previewUrl"):
            return data["results"][0]["previewUrl"]
        return None
    except Exception:
        return None
