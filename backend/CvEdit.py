import cv2
import numpy as np
import json
import os
from groq import Groq
from dotenv import load_dotenv
import os as _os

load_dotenv()
groq_client = Groq(api_key=_os.getenv("GROQ_API_KEY"))
MODEL = "llama-3.3-70b-versatile"
from PIL import Image

# ===== COMMAND LIBRARY =====
# Controlled set of OpenCV operations Gemini can choose from.
# Prevents hallucination by constraining suggestions to real functions.

EDIT_LIBRARY = {
    "adjust_brightness": {
        "description": "Adjust image brightness",
        "params": {"value": "int, -100 to 100, 0 = no change"}
    },
    "adjust_contrast": {
        "description": "Adjust image contrast",
        "params": {"value": "float, 0.5 to 3.0, 1.0 = no change"}
    },
    "adjust_saturation": {
        "description": "Adjust color saturation",
        "params": {"value": "float, 0.0 to 3.0, 1.0 = normal"}
    },
    "adjust_hue": {
        "description": "Shift hue in degrees",
        "params": {"value": "int, -90 to 90, 0 = no change"}
    },
    "adjust_temperature": {
        "description": "Warm (+positive) or cool (-negative) the image",
        "params": {"value": "int, -50 to 50"}
    },
    "gaussian_blur": {
        "description": "Soft blur (must use odd kernel size)",
        "params": {"kernel_size": "int, odd number, 3 to 31"}
    },
    "sharpen": {
        "description": "Sharpen the image using an unsharp mask kernel",
        "params": {"strength": "float, 0.5 to 2.0, 1.0 = standard"}
    },
    "vignette": {
        "description": "Darken edges toward the center of the image",
        "params": {"intensity": "float, 0.0 to 1.0"}
    }
}

ALLOWED_EDIT_COMMANDS = list(EDIT_LIBRARY.keys())


def get_edit_commands(caption, vibe):
    """
    Send the caption + vibe to Groq and get back a list of edit commands.
    Groq picks from EDIT_LIBRARY. No crop or resize — filters and adjustments only.

    Args:
        caption (str): The BLIP-generated caption.
        vibe (str): The chosen vibe for image editing.

    Returns:
        list[dict]: List of {"command": str, "params": dict} objects.
    """
    library_json = json.dumps(EDIT_LIBRARY, indent=2)

    prompt = (
        f"You are an image editing assistant. Given the image caption: "
        f"\"{caption}\", suggest 2-5 filter adjustments to give the image a '{vibe}' vibe.\n\n"
        f"Do NOT crop, resize, or change the image dimensions in any way.\n"
        f"Only use these commands (all are filter/colour adjustments):\n{library_json}\n\n"
        f"Rules:\n"
        f"- Return ONLY a JSON array, no markdown fences, no explanation.\n"
        f"- Each element: {{\"command\": \"name\", \"params\": {{...}}}}\n"
        f"- Do NOT use crop_center or resize.\n"
        f"- Ensure all parameter values are within the specified ranges.\n"
        f"- For kernel_size, always use an odd number."
    )

    response = groq_client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}]
    )
    raw = response.choices[0].message.content.strip()

    if raw.startswith("```"):
        raw = raw.split("\n", 1)[1]
        raw = raw.rsplit("```", 1)[0]

    raw = raw.strip()

    try:
        commands = json.loads(raw)
    except json.JSONDecodeError:
        print(f"❌ Failed to parse Groq response as JSON:\n{raw}")
        return []

    # Filter out any crop/resize commands Gemini might suggest
    forbidden = {"crop_center", "resize"}
    commands = [cmd for cmd in commands if cmd.get("command", "") not in forbidden]

    return commands


def pil_to_cv2(pil_image):
    """Convert a PIL Image (RGB) to an OpenCV image (BGR numpy array)."""
    img = np.array(pil_image)
    return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)


def cv2_to_pil(cv2_image):
    """Convert an OpenCV image (BGR numpy array) to a PIL Image (RGB)."""
    rgb = cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)


def center_crop(img, ratio_str):
    """Center crop an OpenCV image to the given aspect ratio string like '4:5'."""
    h, w = img.shape[:2]
    rw, rh = map(int, ratio_str.split(":"))
    target_ratio = rw / rh
    current_ratio = w / h

    if current_ratio > target_ratio:
        # Image is wider than target — crop width
        new_w = int(h * target_ratio)
        x1 = (w - new_w) // 2
        img = img[:, x1:x1 + new_w]
    else:
        # Image is taller than target — crop height
        new_h = int(w / target_ratio)
        y1 = (h - new_h) // 2
        img = img[y1:y1 + new_h, :]

    return img


def apply_vignette(img, intensity):
    """Apply a vignette (darkened edges) effect to an OpenCV image."""
    h, w = img.shape[:2]
    X = cv2.getGaussianKernel(w, w * (1 - intensity))
    Y = cv2.getGaussianKernel(h, h * (1 - intensity))
    kernel = Y * X.T
    mask = kernel / kernel.max()
    vignette = np.copy(img)
    for i in range(3):
        vignette[:, :, i] = (vignette[:, :, i] * mask).astype(np.uint8)
    return vignette


def apply_edit_commands(pil_image, commands):
    """
    Apply a list of edit commands to a PIL Image and return the edited PIL Image.

    Args:
        pil_image (PIL.Image): The original image.
        commands (list[dict]): List of {"command": str, "params": dict} objects.

    Returns:
        PIL.Image: The edited image.
    """
    if not commands:
        print("⚠️  No edit commands to apply.")
        return pil_image

    img = pil_to_cv2(pil_image)

    for i, cmd in enumerate(commands, start=1):
        name = cmd.get("command", "")
        params = cmd.get("params", {})

        if name not in EDIT_LIBRARY:
            print(f"  ⚠️  Skipping unknown command: {name}")
            continue

        try:
            if name == "adjust_brightness":
                val = int(params["value"])
                img = cv2.convertScaleAbs(img, alpha=1, beta=val)

            elif name == "adjust_contrast":
                val = float(params["value"])
                img = cv2.convertScaleAbs(img, alpha=val, beta=0)

            elif name == "adjust_saturation":
                val = float(params["value"])
                hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float64)
                hsv[:, :, 1] *= val
                hsv[:, :, 1] = np.clip(hsv[:, :, 1], 0, 255)
                img = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

            elif name == "adjust_hue":
                val = int(params["value"])
                hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.int16)
                hsv[:, :, 0] = (hsv[:, :, 0] + val // 2) % 180
                img = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

            elif name == "adjust_temperature":
                val = int(params["value"])
                b, g, r = cv2.split(img)
                if val > 0:
                    r = cv2.add(r, val)
                    b = cv2.subtract(b, val // 2)
                else:
                    b = cv2.add(b, abs(val))
                    r = cv2.subtract(r, abs(val) // 2)
                img = cv2.merge([b, g, r])

            elif name == "gaussian_blur":
                k = int(params["kernel_size"])
                if k % 2 == 0:
                    k += 1
                img = cv2.GaussianBlur(img, (k, k), 0)

            elif name == "sharpen":
                strength = float(params["strength"])
                kernel = np.array([[-1, -1, -1],
                                   [-1, 9, -1],
                                   [-1, -1, -1]]) * strength
                kernel[1, 1] = 1 + 8 * strength  # center pixel: original + sum of negated neighbors
                img = cv2.filter2D(img, -1, kernel)

            elif name == "vignette":
                intensity = float(params["intensity"])
                img = apply_vignette(img, intensity)

            print(f"  ✅ [{i}/{len(commands)}] Applied {name} with {params}")

        except Exception as e:
            print(f"  ❌ [{i}/{len(commands)}] Failed to apply {name}: {str(e)}")

    return cv2_to_pil(img)


def show_before_after(original, edited, output_path):
    """
    Save the edited image and display a before/after comparison window.

    Args:
        original (PIL.Image): The original image.
        edited (PIL.Image): The edited image.
        output_path (str): File path to save the edited image.
    """
    # Save edited image
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    edited.save(output_path)
    print(f"\n📁 Saved edited image to: {output_path}")

    # Create side-by-side comparison
    orig_arr = np.array(original)
    edit_arr = np.array(edited)

    # Resize both to same height if they differ (e.g. after crop)
    if orig_arr.shape[0] != edit_arr.shape[0]:
        target_h = min(orig_arr.shape[0], edit_arr.shape[0])
        orig_arr = cv2.resize(orig_arr, (int(orig_arr.shape[1] * target_h / orig_arr.shape[0]), target_h))
        edit_arr = cv2.resize(edit_arr, (int(edit_arr.shape[1] * target_h / edit_arr.shape[0]), target_h))

    comparison = np.hstack([orig_arr, edit_arr])

    # Add labels
    label_h = 40
    label_bar = np.zeros((label_h, comparison.shape[1], 3), dtype=np.uint8)
    label_bar[:] = (30, 30, 30)  # dark gray background
    mid = comparison.shape[1] // 2
    cv2.putText(label_bar, "BEFORE", (mid - 120, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
    cv2.putText(label_bar, "AFTER", (mid + 30, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
    comparison = np.vstack([label_bar, comparison])

    # Scale down if too tall for screen
    max_h = 900
    if comparison.shape[0] > max_h:
        scale = max_h / comparison.shape[0]
        comparison = cv2.resize(comparison, None, fx=scale, fy=scale)

    # Convert RGB → BGR for cv2 display
    comparison_bgr = cv2.cvtColor(comparison, cv2.COLOR_RGB2BGR)

    cv2.imshow("Before | After", comparison_bgr)
    print("👁️  Press any key in the image window to continue...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def run_cv_edit(image, caption, vibe, output_path="output/edited_image.jpg"):
    """
    Standalone entry point: get edit commands from Groq, apply them, show result.

    Args:
        image (PIL.Image): The original image.
        caption (str): The BLIP caption.
        vibe (str): The chosen vibe.
        output_path (str): Where to save the edited image.
    """
    print(f"\n🎨 Requesting '{vibe}' edit commands from Groq...")
    commands = get_edit_commands(caption, vibe)

    if not commands:
        print("⚠️  No commands returned. Skipping image edits.")
        return image

    print(f"📋 Suggested edits ({len(commands)} commands):")
    for i, cmd in enumerate(commands, start=1):
        print(f"  {i}. {cmd['command']}({cmd['params']})")

    print("\n⚙️  Applying edits...")
    edited = apply_edit_commands(image, commands)

    show_before_after(image, edited, output_path)

    return edited
