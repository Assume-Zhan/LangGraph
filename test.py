
# Import Pytorch library
import torch
import torch.nn as nn

# Import OpenAI CLIP
import clip

# Import necessary libraries
import numpy as np
from PIL import Image

def prepare_model(name='ViT-B/32', device='cuda'):

    # Load the model
    print(f"=========== Loading model {name} with device {device} ===========")
    model, preprocess = clip.load(name, device=device)

    return model, preprocess

def encode_image(image_path, model, preprocess):

    # Load all of the images (which is in the list)
    original_images = [Image.open(image).convert("RGB") for image in image_path]

    # Preprocess the images
    images = torch.stack([preprocess(image) for image in original_images]).to(device)

    # Encode the images
    with torch.no_grad():
        image_features = model.encode_image(images)

    return image_features

def encode_text(text, model):

    # Encode the text
    text_input = clip.tokenize(text).to(device)
    with torch.no_grad():
        text_features = model.encode_text(text_input)

    return text_features

if __name__ == '__main__':

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Load the model
    model, preprocess = prepare_model(name='ViT-B/32', device=device)

    # Set the image path
    image_path = ['./data/img/bed.png', './data/img/barbell.png', './data/img/blue_chair.png', './data/img/org_chair_table.png']

    # Encode the image
    image_features = encode_image(image_path, model, preprocess)

    print(f"Image features shape: {image_features.shape}")

    # Set the text
    text = ["Go to the bed", "Go to the barbell", "Go to the chairs with blue color", "Go to the chairs with yellow color"]

    # Encode the text
    text_features = encode_text(text, model)

    print(f"Text features shape: {text_features.shape}")

    # Calculate the similarity
    similarity = (100.0 * image_features @ text_features.T)

    # Print the similarity
    print(f"Similarity: {similarity}")

    # Get the index of the maximum similarity and print the text
    max_index = torch.argmax(similarity, dim=1)
    for i in range(len(max_index)):
        print(f"Image {i} is similar to text: {text[max_index[i]]}")

    print("=========== Done ===========")