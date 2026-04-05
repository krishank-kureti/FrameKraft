import io
import base64
from PIL import Image


def bytes_to_image(data):
    """
    Convert raw image bytes to a PIL Image.

    Args:
        data (bytes): Raw image data.

    Returns:
        PIL.Image.Image: The loaded image in RGB mode.
    """
    return Image.open(io.BytesIO(data)).convert("RGB")


def image_to_base64(pil_image, fmt="JPEG"):
    """
    Encode a PIL Image to a base64 data URI string.

    Args:
        pil_image (PIL.Image): The image to encode.
        fmt (str): Output format ('JPEG' or 'PNG').

    Returns:
        str: Data URI string like 'data:image/jpeg;base64,...'
    """
    buf = io.BytesIO()
    if pil_image.mode == "RGBA" and fmt == "JPEG":
        pil_image = pil_image.convert("RGB")
    pil_image.save(buf, format=fmt)
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    mime = "image/jpeg" if fmt == "JPEG" else "image/png"
    return f"data:{mime};base64,{b64}"
