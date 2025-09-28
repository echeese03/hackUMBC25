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
from pydantic import BaseModel
from torchvision.models import ResNet18_Weights

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

class ImagePath(BaseModel):
    image_path: str

@app.post("/classify/")
async def classify_image(image_path: ImagePath):
    img = Image.open(image_path.image_path)
    img_tf = app.state.transform(img)
    img_output = app.state.model(img_tf.unsqueeze(0))
    classification = app.state.classes[np.argmax(img_output.detach().numpy(), axis=1)[0]]
    print(classification)
    return classification

def classify_image(image_path, model):
    img = Image.open(image_path)
    img_tf = init_transforms()(img)
    img_output = model(img_tf.unsqueeze(0))
    return init_classes()[np.argmax(img_output.detach().numpy(), axis=1)[0]]

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
    classification = classify_image(image_path, SpecializedRecyclableClassifier)
    print("Classification: ", classification)