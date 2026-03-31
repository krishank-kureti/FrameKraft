from PIL import Image
import requests
from transformers import CLIPProcessor, CLIPModel
import torch

def load_clip_model():
    """Load and return CLIP model and processor"""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    print("Loading CLIP model...")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    model.to(device)

    print("CLIP model loaded successfully!\n")
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

def classify_image(image, labels, processor, model, device):
    """Perform zero-shot classification on an image"""
    inputs = processor(text=labels, images=image, return_tensors="pt", padding=True).to(device)

    with torch.no_grad():
        outputs = model(**inputs)
        logits_per_image = outputs.logits_per_image  # this is the image-text similarity score
        probs = logits_per_image.softmax(dim=1)  # we can take the softmax to get the label probabilities

    return probs[0]

def run_clip(image_sources, text_labels):
    """Main function to run CLIP classification on multiple images"""
    processor, model, device = load_clip_model()

    # Process each image source
    for idx, source in enumerate(image_sources, start=1):
        print(f"\n{'='*60}")
        print(f"Processing Image {idx}: {source[:50]}..." if len(source) > 50 else f"Processing Image {idx}: {source}")
        print('='*60)

        try:
            image = load_image(source)

            # Perform zero-shot classification
            probabilities = classify_image(image, text_labels, processor, model, device)

            print("✅ CLIP Zero-Shot Classification Results:")
            # Sort by probability and show top predictions
            top_indices = torch.topk(probabilities, len(text_labels)).indices

            for i, idx in enumerate(top_indices, start=1):
                label = text_labels[idx]
                prob = probabilities[idx].item() * 100
                print(f"{i}. {label}: {prob:.1f}%")

            print(f"\n🎯 Top prediction: '{text_labels[top_indices[0]]}' with {probabilities[top_indices[0]].item()*100:.1f}% confidence")

        except Exception as e:
            print(f"❌ Error processing image: {str(e)}")

    print(f"\n{'='*60}")
    print("CLIP processing complete!")
    print('='*60)