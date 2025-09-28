import os
import torch
import torchvision.transforms as transforms
from efficientnet_pytorch import EfficientNet
import torch.nn as nn
from PIL import Image
from pathlib import Path
import sys

def main():
    ROOT_DIR = Path(__file__).resolve().parent
    # get image path from argument
    source_img = ROOT_DIR / 'test_imgs' / 'can.jpg'
    if len(sys.argv) > 1:
        # assuming path input (not just filename)
        source_img = sys.argv[1]
    else:
        print("No image provided, using test image.")
    # ===============================
    # STEP 1: Setup Device
    # ===============================
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Using device:", device)

    # ===============================
    # STEP 2: Define Class Names and Number of Classes
    # ===============================
    class_names = [
        "Aerosols", "Aluminum can", "Aluminum caps", "Cardboard", "Cellulose", "Ceramic", 
        "Combined plastic", "Container for household chemicals", "Disposable tableware", 
        "Electronics", "Foil", "Furniture", "Glass bottle", "Iron utensils", "Liquid", 
        "Metal shavings", "Milk bottle", "Organic", "Paper bag", "Paper cups", "Paper shavings", 
        "Paper", "Papier mache", "Plastic bag", "Plastic bottle", "Plastic can", "Plastic canister", 
        "Plastic caps", "Plastic cup", "Plastic shaker", "Plastic shavings", "Plastic toys", 
        "Postal packaging", "Printing industry", "Scrap metal", "Stretch film", "Tetra pack", 
        "Textile", "Tin", "Unknown plastic", "Wood", "Zip plastic bag", "Ramen Cup", "Food Packet"
    ]

    nc = len(class_names)  # Number of classes

    # ===============================
    # STEP 3: Load the Model
    # ===============================
    model = EfficientNet.from_pretrained('efficientnet-b0')
    model._fc = nn.Linear(model._fc.in_features, nc)  # Adjust for 44 classes
    model_path = ROOT_DIR / 'yolo' / 'best_model.pth'
    model.load_state_dict(torch.load(model_path))  # Load your trained model
    model.to(device)
    model.eval()  # Set the model to evaluation mode

    # ===============================
    # STEP 4: Define Transforms
    # ===============================
    transform = transforms.Compose([
        transforms.Resize((300, 300)),
        transforms.ToTensor(),
    ])

    # ===============================
    # STEP 5: Function to Make Predictions
    # ===============================
    def predict(image_path):
        # Load and preprocess the image
        image = Image.open(image_path).convert("RGB")
        image = transform(image).unsqueeze(0)  # Add batch dimension
        image = image.to(device)

        # Make prediction
        with torch.no_grad():  # Disable gradient calculations for inference
            outputs = model(image)
            probabilities = torch.softmax(outputs, dim=1)  # Convert logits to probabilities

            # Get the predicted class and its confidence
            confidences, predicted_class = torch.max(probabilities, 1)
            predicted_class = predicted_class.item()  # Get the class index as an integer
            confidence_score = confidences.item()  # Get the confidence score as a float

        return class_names[predicted_class], confidence_score

    predicted_class, confidence_score = predict(source_img)
    print(f'Predicted class: {predicted_class}, Confidence: {confidence_score:.2f}')

if __name__ == "__main__":
    main()