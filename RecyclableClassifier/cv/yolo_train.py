import numpy as np
from ultralytics import YOLO
import torch

if __name__ == "__main__":
    torch.cuda.empty_cache()
    model = YOLO('yolov8s.pt')
    model.to('cuda')

    # Train using the single most idle GPU
    # results = model.train(data="coco8.yaml", epochs=100, imgsz=640, device=-1)

    # # Train using the two most idle GPUs
    # results = model.train(data="coco8.yaml", epochs=100, imgsz=640, device=[-1, -1])
    # Load a model
    results = model.train(data="C:\\Users\\ethan\\.cache\\kagglehub\\datasets\\spellsharp\\garbage-data\\versions\\14\\data.yaml", 
                          epochs=30, imgsz=640, device=[0], batch=4)
    
    # results = model.train(data="C:\\Users\\ethan\\.cache\\kagglehub\\datasets\\spellsharp\\garbage-data\\versions\\14\\data.yaml", 
    #                       epochs=30, imgsz=640, device=[0], batch=4, resume=True)  # Adjust batch as needed

    # Train using the single most idle GPU
    # results = model.train(data="C:\\Users\\ethan\\.cache\\kagglehub\\datasets\\spellsharp\\garbage-data\\versions\\14\\data.yaml", epochs=30, imgsz=640, device=[-1, -1])