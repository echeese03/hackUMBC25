from contextlib import asynccontextmanager
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

import torch
from torch import nn
from torchvision import transforms
import torchvision.models as models
from torch.nn import Module
from PIL import Image
import numpy as np
import sys
import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from torchvision.models import ResNet18_Weights
import base64
import io

styrofoam_classes = ['styrofoam_cups', 'styrofoam_food_containers']
metal_classes = ['aerosol_cans', 'aluminum_food_cans', 'aluminum_soda_cans', 'steel_food_cans']
cardboard_classes = ['cardboard_boxes', 'cardboard_packaging']
clothing_classes = ['clothing', 'shoes']
food_waste_classes = ['coffee_grounds', 'eggshells', 'food_waste', 'tea_bags']
glass_classes = ['glass_beverage_bottles', 'glass_cosmetic_containers', 'glass_food_jars']
paper_classes = ['magazines', 'newspaper', 'office_paper', 'paper_cups']
plastic_classes = ['plastic_cup_lids', 'plastic_detergent_bottles', 'plastic_food_containers', 'plastic_shopping_bags', 'plastic_soda_bottles', 'plastic_straws', 'plastic_trash_bags', 'plastic_water_bottles', 'disposable_plastic_cutlery']

class CustomResNet18(Module):
    def __init__(self):
        super().__init__()
        resnet_model = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        resnet_model.fc = nn.Linear(resnet_model.fc.in_features, 30)
        self.model = resnet_model

    def forward(self, image):
        x = self.model(image)
        return x

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Initializing Specialized Recyclable Classifier Model")
    app.state.model = init_model()
    app.state.transform = init_transforms()
    app.state.classes = init_classes()

    yield

    print("Shutting Down Specialized Recyclable Classifier Model")

app = FastAPI(lifespan=lifespan)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def init_transforms():
    image_tf = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    return image_tf

def init_model():
    model = CustomResNet18()
    model.load_state_dict(torch.load("Resnet18Recyclable.pth", weights_only=True, map_location=torch.device('cpu')))
    model.eval()
    return model

def init_classes():
    classes = ['aerosol_cans', 'aluminum_food_cans', 'aluminum_soda_cans', 'cardboard_boxes', 'cardboard_packaging', 'clothing', 'coffee_grounds', 'disposable_plastic_cutlery', 'eggshells', 'food_waste', 'glass_beverage_bottles', 'glass_cosmetic_containers', 'glass_food_jars', 'magazines', 'newspaper', 'office_paper', 'paper_cups', 'plastic_cup_lids', 'plastic_detergent_bottles', 'plastic_food_containers', 'plastic_shopping_bags', 'plastic_soda_bottles', 'plastic_straws', 'plastic_trash_bags', 'plastic_water_bottles', 'shoes', 'steel_food_cans', 'styrofoam_cups', 'styrofoam_food_containers', 'tea_bags']
    return classes

# Updated model to accept base64 image
class ImageData(BaseModel):
    image: str  # Base64 encoded image

@app.post("/classify/")
async def classify(image_data: ImageData):
    try:
        # Extract base64 data
        if image_data.image.startswith('data:image'):
            # Remove the data URL prefix
            base64_str = image_data.image.split(',')[1]
        else:
            base64_str = image_data.image
            
        # Decode base64 to bytes
        image_bytes = base64.b64decode(base64_str)
        img = Image.open(io.BytesIO(image_bytes))
        
        # Convert RGBA to RGB if necessary
        if img.mode == 'RGBA':
            img = img.convert('RGB')
        
        # Apply transforms
        img_tf = app.state.transform(img)
        
        # Get prediction
        img_output = app.state.model(img_tf.unsqueeze(0))
        classification = app.state.classes[np.argmax(img_output.detach().numpy(), axis=1)[0]]
        
        # Generalize classification
        generalized_classification = generalize_classification(classification)
        
        return {
            "classification": generalized_classification,
            "specific_classification": classification,
            "success": True
        }
        
    except Exception as e:
        return {"error": str(e), "success": False}

def generalize_classification(classification):
    """Generalize specific classifications into broader categories"""
    if classification in styrofoam_classes:
        return "styrofoam"
    elif classification in metal_classes:
        return "metal"
    elif classification in cardboard_classes:
        return "cardboard"
    elif classification in clothing_classes:
        return "clothing"
    elif classification in food_waste_classes:
        return "food_waste"
    elif classification in glass_classes:
        return "glass"
    elif classification in paper_classes:
        return "paper"
    elif classification in plastic_classes:
        return "plastic"
    else:
        return classification

# For standalone script usage - now also accepts base64 images
def classify_image_from_file(image_path, model, generalize=False):
    """Helper function for command line usage"""
    img = Image.open(image_path)
    if img.mode == 'RGBA':
        img = img.convert('RGB')
    
    img_tf = init_transforms()(img)
    img_output = model(img_tf.unsqueeze(0))
    classification = init_classes()[np.argmax(img_output.detach().numpy(), axis=1)[0]]
    
    if generalize:
        classification = generalize_classification(classification)
    
    return classification

# Main function to run the script standalone
if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python SpecializedRecyclableClassifier.py <image_path>")
        sys.exit(1)

    image_path = sys.argv[1]

    if not os.path.isfile(image_path):
        print(f"Error: Image file '{image_path}' not found.")
        sys.exit(1)
    
    # Load the model
    SpecializedRecyclableClassifier = init_model()

    # Classify the image
    classification = classify_image_from_file(image_path, SpecializedRecyclableClassifier, generalize=True)
    print("Classification: ", classification)