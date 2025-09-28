from contextlib import asynccontextmanager
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

import torch
from ultralytics import YOLO
from torch import nn
from torchvision import transforms
import torchvision.models as models
from torch.nn import Module
from PIL import Image
from PIL import ImageDraw
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
    app.state.YOLOmodel = init_yolo_model()
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

def init_yolo_model():
    model = YOLO("model.pt", task="detect")
    return model

def init_classes():
    classes = ['aerosol_cans', 'aluminum_food_cans', 'aluminum_soda_cans', 'cardboard_boxes', 'cardboard_packaging', 'clothing', 'coffee_grounds', 'disposable_plastic_cutlery', 'eggshells', 'food_waste', 'glass_beverage_bottles', 'glass_cosmetic_containers', 'glass_food_jars', 'magazines', 'newspaper', 'office_paper', 'paper_cups', 'plastic_cup_lids', 'plastic_detergent_bottles', 'plastic_food_containers', 'plastic_shopping_bags', 'plastic_soda_bottles', 'plastic_straws', 'plastic_trash_bags', 'plastic_water_bottles', 'shoes', 'steel_food_cans', 'styrofoam_cups', 'styrofoam_food_containers', 'tea_bags']
    return classes

class ImagePath(BaseModel):
    image_path: str

@app.post("/classify/")
async def classify_image(image_path: ImagePath):
    img = Image.open(image_path.image_path)

    # YOLO Detection
    res = app.state.YOLOmodel.predict(img, conf=0.4)
    boxes = res[0].boxes
    img_array = np.array(img)
    img_height, img_width = img_array.shape[:2]

    # Parameters for cropping
    PADDING = 0  # pixels of padding around each box
    FILL_COLOR = (0, 0, 0)  # RGB color for out-of-bounds pixels

    # handle different image formats
    if len(img_array.shape) == 3:   #RGB
        num_channels = img_array.shape[2]
        if num_channels == 4:       # RGBA
            FILL_COLOR = (*FILL_COLOR, 255)
        elif num_channels != 3:     # other
            FILL_COLOR = tuple([FILL_COLOR[0]] * num_channels)

    box_imgs = []
    if boxes is not None and len(boxes) > 0:
        for idx, box in enumerate(boxes):
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)

            box_width = x2 - x1
            box_height = y2 - y1

            # find the larger dimension and add PADDING to create square
            max_dim = max(box_width, box_height)
            square_size = max_dim + 2 * PADDING

            # find center of the bounding box
            box_center_x = (x1 + x2) // 2
            box_center_y = (y1 + y2) // 2

            # calculate square crop coordinates
            crop_x1 = box_center_x - square_size // 2
            crop_y1 = box_center_y - square_size // 2
            crop_x2 = crop_x1 + square_size
            crop_y2 = crop_y1 + square_size

            # Create a square image filled with the fill color
            if len(img_array.shape) == 3:
                cropped = np.full((square_size, square_size, img_array.shape[2]),
                                  FILL_COLOR, dtype=np.uint8)
            else:
                cropped = np.full((square_size, square_size),
                                  FILL_COLOR[0], dtype=np.uint8)

            # calculate the source and destination regions for copying
            src_x1 = max(0, crop_x1)
            src_y1 = max(0, crop_y1)
            src_x2 = min(img_width, crop_x2)
            src_y2 = min(img_height, crop_y2)

            dst_x1 = src_x1 - crop_x1
            dst_y1 = src_y1 - crop_y1
            dst_x2 = dst_x1 + (src_x2 - src_x1)
            dst_y2 = dst_y1 + (src_y2 - src_y1)

            # copy bounding box to new image
            cropped[dst_y1:dst_y2, dst_x1:dst_x2] = img_array[src_y1:src_y2, src_x1:src_x2]

            # convert back to PIL Image and save
            cropped_pil = Image.fromarray(cropped)

            # get class name if available
            class_id = int(box.cls[0])
            class_name = app.state.YOLOmodel.names.get(class_id, f"class_{class_id}")
            confidence_score = box.conf[0]

            final_box_img = None
            # Convert RGBA to RGB before saving as JPEG (JPEG doesn't support alpha)
            if cropped_pil.mode == 'RGBA':
                rgb_img = Image.new('RGB', cropped_pil.size, (255, 255, 255))
                rgb_img.paste(cropped_pil, mask=cropped_pil.split()[3])  # Use alpha as mask
                final_box_img = rgb_img
            else:
                final_box_img = cropped_pil



            print(f"  Original box: [{x1}, {y1}, {x2}, {y2}] (w:{box_width}, h:{box_height})")
            print(f"  Square crop: [{crop_x1}, {crop_y1}, {crop_x2}, {crop_y2}] (size:{square_size}x{square_size})")
            print(f"  Class: {class_name}, Confidence: {confidence_score:.2f}")

    else:
        print("No boxes detected!")

    # END YOLO Detection
    classifications = []
    for box_img in box_imgs:
        img_tf = app.state.transform(box_img)
        img_output = app.state.model(img_tf.unsqueeze(0))
        classification = app.state.classes[np.argmax(img_output.detach().numpy(), axis=1)[0]]
        print(classification)
    return classifications

    #img_tf = app.state.transform(img)
    #img_output = app.state.model(img_tf.unsqueeze(0))
    #classification = app.state.classes[np.argmax(img_output.detach().numpy(), axis=1)[0]]
    #print(classification)
    #return classification

def classify_image(image_path, model, YOLOmodel):
    img = Image.open(image_path)

    # YOLO Detection
    res = YOLOmodel.predict(img, conf=0.45)
    boxes = res[0].boxes
    img_array = np.array(img)
    img_height, img_width = img_array.shape[:2]

    # Parameters for cropping
    PADDING = 0  # pixels of padding around each box
    FILL_COLOR = (255, 255, 255)  # RGB color for out-of-bounds pixels

    # handle different image formats
    if len(img_array.shape) == 3:   #RGB
        num_channels = img_array.shape[2]
        if num_channels == 4:       # RGBA
            FILL_COLOR = (*FILL_COLOR, 255)
        elif num_channels != 3:     # other
            FILL_COLOR = tuple([FILL_COLOR[0]] * num_channels)

    box_imgs = []
    if boxes is not None and len(boxes) > 0:
        for idx, box in enumerate(boxes):
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)

            box_width = x2 - x1
            box_height = y2 - y1

            # find the larger dimension and add PADDING to create square
            max_dim = max(box_width, box_height)
            square_size = max_dim + 2 * PADDING

            # find center of the bounding box
            box_center_x = (x1 + x2) // 2
            box_center_y = (y1 + y2) // 2

            # calculate square crop coordinates
            crop_x1 = box_center_x - box_width // 2
            crop_y1 = box_center_y - box_height // 2
            crop_x2 = crop_x1 + box_width
            crop_y2 = crop_y1 + box_height

            # Create a square image filled with the fill color
            if len(img_array.shape) == 3:
                cropped = np.full((square_size, square_size, img_array.shape[2]),
                                  FILL_COLOR, dtype=np.uint8)
            else:
                cropped = np.full((square_size, square_size),
                                  FILL_COLOR[0], dtype=np.uint8)

            # calculate the source and destination regions for copying
            src_x1 = max(0, crop_x1)
            src_y1 = max(0, crop_y1)
            src_x2 = min(img_width, crop_x2)
            src_y2 = min(img_height, crop_y2)

            dst_x1 = src_x1 - crop_x1
            dst_y1 = src_y1 - crop_y1
            dst_x2 = dst_x1 + (src_x2 - src_x1)
            dst_y2 = dst_y1 + (src_y2 - src_y1)

            # copy bounding box to new image
            cropped[dst_y1:dst_y2, dst_x1:dst_x2] = img_array[src_y1:src_y2, src_x1:src_x2]

            # convert back to PIL Image and save
            cropped_pil = Image.fromarray(cropped)

            # get class name if available
            class_id = int(box.cls[0])
            class_name = YOLOmodel.names.get(class_id, f"class_{class_id}")
            confidence_score = box.conf[0]

            final_box_img = None
            # Convert RGBA to RGB before saving as JPEG (JPEG doesn't support alpha)
            if cropped_pil.mode == 'RGBA':
                rgb_img = Image.new('RGB', cropped_pil.size, (255, 255, 255))
                rgb_img.paste(cropped_pil, mask=cropped_pil.split()[3])  # Use alpha as mask
                final_box_img = rgb_img
            else:
                final_box_img = cropped_pil

            filename = f"box_{idx:03d}_{class_name}_{confidence_score:.2f}.jpg"
            save_path = filename
            final_box_img.save(save_path)
            box_imgs.append(final_box_img)

            box_imgs.append(final_box_img)

            print(f"  Original box: [{x1}, {y1}, {x2}, {y2}] (w:{box_width}, h:{box_height})")
            print(f"  Square crop: [{crop_x1}, {crop_y1}, {crop_x2}, {crop_y2}] (size:{square_size}x{square_size})")
            print(f"  Class: {class_name}, Confidence: {confidence_score:.2f}")

    else:
        print("No boxes detected!")

    # END YOLO Detection
    classifications = []
    for box_img in box_imgs:
        img_tf = init_transforms()(box_img)
        img_output = model(img_tf.unsqueeze(0))
        classification = init_classes()[np.argmax(img_output.detach().numpy(), axis=1)[0]]
        print(classification)
    return classifications
    #img = Image.open(image_path)
    #img_tf = init_transforms()(img)
    #img_output = model(img_tf.unsqueeze(0))
    #return init_classes()[np.argmax(img_output.detach().numpy(), axis=1)[0]]

# Main function to run the script standalone
if __name__ == '__main__':

    if len(sys.argv) < 2:
        print("Usage: python IntegratedRecyclableClassifier.py <image_path>")
        sys.exit(1)

    image_path = sys.argv[1]

    if not os.path.isfile(image_path):
        print(f"Error: Image file '{image_path}' not found.")
        sys.exit(1)
    # Load the model
    SpecializedRecyclableClassifier = init_model()
    YOLOModel = init_yolo_model()

    # Classify the image
    classification = classify_image(image_path, SpecializedRecyclableClassifier, YOLOModel)
    print("Classification: ", classification)