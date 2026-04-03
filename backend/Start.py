from Clip import load_clip_model, load_image as load_image_clip, classify_image
from Blip import load_blip_model, load_image as load_image_blip, generate_captions
from CvEdit import run_cv_edit
from MusicRag import recommend_songs
import torch
from groq import Groq
from dotenv import load_dotenv
import os

load_dotenv()
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
MODEL = "llama-3.3-70b-versatile"

# ===== IMAGE SOURCES CONFIGURATION =====
# Add your image sources here (URLs and/or local paths)
image_sources = [
    # Example URLs
    "https://images.unsplash.com/photo-1590107110321-cca1bc307cc7?q=80&w=987&auto=format&fit=crop&ixlib=rb-4.1.0&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D"
]
# ===== END IMAGE SOURCES CONFIGURATION =====

def pick_mood():
    """Let the user specify the mood/tone for their caption."""
    print("\n🎭 What mood should the captions have?")
    print("  (e.g. playful, melancholic, dramatic, mysterious, romantic, energetic...)")
    while True:
        mood = input("\nEnter a mood: ").strip()
        if mood:
            return mood
        print("Please enter a mood.")

def pick_vibe():
    """Let the user specify a vibe for image editing."""
    print("\n🎨 What vibe should the image editing have?")
    print("  (e.g. cinematic, warm, cool, moody, vintage, high contrast...)")
    while True:
        vibe = input("\nEnter a vibe: ").strip()
        if vibe:
            return vibe
        print("Please enter a vibe.")

def stylize_caption(caption, mood):
    """
    Generate 3 stylized Instagram-style captions using Groq llama-3.3-70b-versatile.

    Args:
        caption (str): The original caption to stylize.
        mood (str): The chosen mood for the caption (e.g. playful, dramatic, mysterious).

    Returns:
        list[str]: 3 stylized captions.
    """
    prompt = (
        f"Given the image caption: '{caption}', generate exactly 3 different Instagram "
        f"post captions with a '{mood}' mood. Each caption should include emojis and "
        f"hashtags. Number them 1, 2, 3. Output only the numbered captions, nothing else."
    )

    response = groq_client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}]
    )
    text = response.choices[0].message.content.strip()

    # Parse numbered lines from the response
    lines = text.split("\n")
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
    """Run the complete five-stage pipeline"""
    print("🚀 Starting FrameKraft Pipeline")
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

            # Step 3: Pick a mood for the caption, then generate captions with Gemini
            best_caption = captions[top_indices[0]]
            mood = pick_mood()
            print(f"\n🎨 Generating 3 '{mood}' captions with Groq...")
            stylized_captions = stylize_caption(best_caption, mood)
            print("\n📱 Your Caption Options:")
            for i, cap in enumerate(stylized_captions, start=1):
                print(f"  {i}. {cap}")

            # Step 4: Pick a vibe for image editing, then apply OpenCV filters
            edit_vibe = pick_vibe()
            output_path = f"output/image{idx}_edited.jpg"
            run_cv_edit(image, best_caption, edit_vibe, output_path)

            # Step 5: Music recommendations
            print("\n🎵 Finding song recommendations...")
            song_recs = recommend_songs(best_caption, mood, edit_vibe, k=5)
            print("\n🎶 Song Recommendations:")
            for i, song in enumerate(song_recs, start=1):
                print(f"  {i}. {song['title']} — {song['artist']}")
                print(f"     Genre: {song['genre']}")
                if song["tags"]:
                    print(f"     Tags: {song['tags']}")
                if song.get("preview_url"):
                    print(f"     🎵 Preview: {song['preview_url']}")
                else:
                    print(f"     🎵 Preview: Not available")

        except Exception as e:
            print(f"❌ Error processing image: {str(e)}")

    print(f"\n{'='*80}")
    print("🎉 FrameKraft Pipeline Complete!")
    print(f"✅ Successfully processed {len(image_sources)} image(s)")
    print('='*80)

if __name__ == "__main__":
    run_pipeline()