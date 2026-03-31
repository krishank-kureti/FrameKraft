from PIL import Image
import requests
from transformers import BlipProcessor, BlipForConditionalGeneration
import torch

def load_blip_model():
    """Load and return BLIP model and processor"""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    print("Loading BLIP model...")
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    model.to(device)

    print("BLIP model loaded successfully!\n")
    return processor, model, device

def load_image(source):
    """Load image from URL or local path"""
    if source.startswith(('http://', 'https://')):
        # Load from URL
        image = Image.open(requests.get(source, stream=True).raw).convert("RGB")
    else:
        # Load from local path
        image = Image.open(source).convert("RGB")
    return image

def generate_captions(image, processor, model, device):
    """Generate captions for an image using BLIP"""
    inputs = processor(image, return_tensors="pt").to(device)

    with torch.no_grad():
        output = model.generate(
            **inputs,
            num_beams=5,
            num_return_sequences=3,
            early_stopping=True
        )

    captions = [processor.decode(out, skip_special_tokens=True).strip() for out in output]
    return captions

def run_blip(image_sources):
    """Main function to run BLIP captioning on multiple images"""
    processor, model, device = load_blip_model()

    # Process each image source
    for idx, source in enumerate(image_sources, start=1):
        print(f"\n{'='*60}")
        print(f"Processing Image {idx}: {source[:50]}..." if len(source) > 50 else f"Processing Image {idx}: {source}")
        print('='*60)

        try:
            image = load_image(source)

            # Generate captions
            captions = generate_captions(image, processor, model, device)

            print("✅ Generated Captions:")
            for i, caption in enumerate(captions, start=1):
                print(f"{i}. {caption}")

        except Exception as e:
            print(f"❌ Error processing image: {str(e)}")

    print(f"\n{'='*60}")
    print("BLIP processing complete!")
    print('='*60)