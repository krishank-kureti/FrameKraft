from Clip import load_clip_model, load_image as load_image_clip, classify_image
from Blip import load_blip_model, load_image as load_image_blip, generate_captions
from CvEdit import run_cv_edit
import torch
import google.generativeai as genai
from dotenv import load_dotenv
import os

load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# ===== IMAGE SOURCES CONFIGURATION =====
# Add your image sources here (URLs and/or local paths)
image_sources = [
    # Example URLs
    "https://images.unsplash.com/photo-1773332598451-8a0a59941912?q=80&w=987&auto=format&fit=crop&ixlib=rb-4.1.0&ixid=M3wxMjA3fDF8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D"
]
# ===== END IMAGE SOURCES CONFIGURATION =====

def pick_vibe():
    """Let the user choose a vibe for their caption."""
    vibes = ["playful", "short and sweet", "chill", "cinematic"]
    print("\n🎭 Choose a vibe for your caption:")
    for i, vibe in enumerate(vibes, start=1):
        print(f"  {i}. {vibe.capitalize()}")
    while True:
        choice = input("\nEnter your choice (1-4): ").strip()
        if choice in ("1", "2", "3", "4"):
            return vibes[int(choice) - 1]
        print("Invalid choice. Please enter 1, 2, 3, or 4.")

def stylize_caption(caption, vibe):
    """
    Generate 3 stylized Instagram-style captions using Google Gemini 2.5 Flash.

    Args:
        caption (str): The original caption to stylize.
        vibe (str): The chosen vibe ('playful', 'short and sweet', 'chill', or 'cinematic').

    Returns:
        list[str]: 3 stylized captions.
    """
    model = genai.GenerativeModel("gemini-2.5-flash")

    prompt = (
        f"Given the image caption: '{caption}', generate exactly 3 different Instagram "
        f"post captions with a '{vibe}' vibe. Each caption should include emojis and "
        f"hashtags. Number them 1, 2, 3. Output only the numbered captions, nothing else."
    )

    response = model.generate_content(prompt)

    # Parse numbered lines from the response
    lines = response.text.strip().split("\n")
    captions = []
    for line in lines:
        line = line.strip()
        if line and line[0].isdigit():
            # Remove leading number/punctuation like "1." or "1 -"
            cleaned = line.lstrip("0123456789").lstrip(".)-– ").strip()
            if cleaned:
                captions.append(cleaned)
    return captions[:3]

def run_pipeline():
    """Run the complete BLIP → CLIP pipeline"""
    print("🚀 Starting BLIP → CLIP Pipeline")
    print(f"📊 Processing {len(image_sources)} image(s)")
    print("="*80)

    # Load both models
    print("Loading models...")
    blip_processor, blip_model, device = load_blip_model()
    clip_processor, clip_model, _ = load_clip_model()

    # Process each image source
    for idx, source in enumerate(image_sources, start=1):
        print(f"\n{'='*80}")
        print(f"🖼️  Processing Image {idx}: {source[:60]}..." if len(source) > 60 else f"🖼️  Processing Image {idx}: {source}")
        print('='*80)

        try:
            # Load image (using BLIP's load_image function)
            image = load_image_blip(source)

            # Step 1: Generate captions using BLIP
            print("\n🔍 Step 1: Generating captions with BLIP...")
            captions = generate_captions(image, blip_processor, blip_model, device)

            print("📝 BLIP Generated Captions:")
            for i, caption in enumerate(captions, start=1):
                print(f"  {i}. {caption}")

            # Step 2: Use BLIP captions as text labels for CLIP classification
            print("\n🎯 Step 2: Using BLIP captions as labels for CLIP classification...")
            probabilities = classify_image(image, captions, clip_processor, clip_model, device)

            print("✅ CLIP Classification Results (using BLIP captions as labels):")
            # Sort by probability and show results
            top_indices = torch.topk(probabilities, len(captions)).indices

            for i, idx in enumerate(top_indices, start=1):
                caption = captions[idx]
                prob = probabilities[idx].item() * 100
                print(f"{i}. {caption}: {prob:.1f}%")

            print(f"\n🏆 Best matching caption: '{captions[top_indices[0]]}' with {probabilities[top_indices[0]].item()*100:.1f}% confidence")

            # Step 3: Pick a vibe, then generate stylized captions using Gemini
            best_caption = captions[top_indices[0]]
            vibe = pick_vibe()
            print(f"\n🎨 Generating 3 '{vibe}' captions with Gemini...")
            stylized_captions = stylize_caption(best_caption, vibe)
            print("\n📱 Your Caption Options:")
            for i, cap in enumerate(stylized_captions, start=1):
                print(f"  {i}. {cap}")

            # Step 4: OpenCV image editing pipeline
            output_path = f"output/image{idx}_edited.jpg"
            run_cv_edit(image, best_caption, vibe, output_path)

        except Exception as e:
            print(f"❌ Error processing image: {str(e)}")

    print(f"\n{'='*80}")
    print("🎉 BLIP → CLIP Pipeline Complete!")
    print(f"✅ Successfully processed {len(image_sources)} image(s)")
    print('='*80)

if __name__ == "__main__":
    run_pipeline()