import io
import base64
from PIL import Image

from Blip import load_blip_model, generate_captions
from Clip import load_clip_model, classify_image
from Start import stylize_caption
from CvEdit import get_edit_commands, apply_edit_commands
from MusicRag import recommend_songs
import torch

_blip_state = None
_clip_state = None


def _load_models():
    """Load BLIP and CLIP models once and cache them."""
    global _blip_state, _clip_state
    if _blip_state is None:
        _blip_state = load_blip_model()
    if _clip_state is None:
        _clip_state = load_clip_model()
    return _blip_state, _clip_state


def _image_to_base64(pil_image, fmt="JPEG"):
    """Encode a PIL Image to a base64 data URI string."""
    buf = io.BytesIO()
    if pil_image.mode == "RGBA" and fmt == "JPEG":
        pil_image = pil_image.convert("RGB")
    pil_image.save(buf, format=fmt)
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    mime = "image/jpeg" if fmt == "JPEG" else "image/png"
    return f"data:{mime};base64,{b64}"


def run_pipeline_for_image(image_stream, mood, vibe):
    """
    Run the full FrameKraft pipeline on an uploaded image.

    Args:
        image_stream: A file-like object (BytesIO) containing image bytes.
        mood (str): User-specified mood for caption stylization.
        vibe (str): User-specified vibe for image editing.

    Returns:
        dict with keys: original_image, edited_image, captions, music
    """
    blip_processor, blip_model, device = _load_models()[0]
    clip_processor, clip_model, _ = _load_models()[1]

    original = Image.open(image_stream).convert("RGB")

    captions = generate_captions(original, blip_processor, blip_model, device)

    probabilities = classify_image(original, captions, clip_processor, clip_model, device)
    top_indices = torch.topk(probabilities, len(captions)).indices
    best_caption = captions[top_indices[0]]

    stylized = stylize_caption(best_caption, mood)

    commands = get_edit_commands(best_caption, vibe)
    edited = apply_edit_commands(original, commands)

    music = recommend_songs(best_caption, mood, vibe, k=5)

    return {
        "original_image": _image_to_base64(original),
        "edited_image": _image_to_base64(edited),
        "captions": stylized,
        "best_caption": best_caption,
        "blip_tags": captions,
        "clip_confidence": float(probabilities[top_indices[0]].item() * 100),
        "edit_commands": commands,
        "music": music
    }
