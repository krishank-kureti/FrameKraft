from PIL import Image
import requests
from transformers import BlipProcessor, BlipForConditionalGeneration
import torch

# Device setup
device = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Using device: {device}")

# Load model (this will download once)
print("Loading model...")
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
model.to(device)

print("Model loaded successfully!\n")

'''
# Test image
image_url = "https://plus.unsplash.com/premium_photo-1684141286798-08c46a23b616?w=900&auto=format&fit=crop&q=60&ixlib=rb-4.1.0&ixid=M3wxMjA3fDB8MHxzZWFyY2h8MXx8ZGFyayUyMGFsbGV5d2F5fGVufDB8fDB8fHww"
image = Image.open(requests.get(image_url, stream=True).raw).convert("RGB")
'''
# Load LOCAL image
image_path = "/Users/krishankkureti/Desktop/My med/Swirling Crimson and Shadow.png"   # <-- put your image here
image = Image.open(image_path).convert("RGB")

# Generate caption
inputs = processor(image, return_tensors="pt").to(device)

with torch.no_grad():
    output = model.generate(**inputs)

caption = processor.decode(output[0], skip_special_tokens=True)

print("✅ Generated Caption:")
print(caption)