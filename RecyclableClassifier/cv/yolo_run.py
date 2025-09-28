from ultralytics import YOLO
from pathlib import Path
import PIL
import sys
import numpy as np

def main():
    CONFIDENCE = .4
    ROOT_DIR = Path(__file__).resolve().parent
    
    # get image path from argument
    source_img = ROOT_DIR / 'test_imgs' / 'can.jpg'
    if len(sys.argv) > 1:
        # assuming path input (not just filename)
        source_img = sys.argv[1]
    else:
        print("No image provided, using test image.")
    # load image as PIL image
    try:
        image = PIL.Image.open(source_img)
    except Exception as ex:
        print("Error occurred while opening the image!")
        raise ex

    # load YOLO model
    model_path = ROOT_DIR / 'yolo' / 'model.pt'
    try:
        model = YOLO(model_path, task="detect")
    except Exception as ex:
        print(f"Error while loading model at {model_path}")
        raise ex

    # run model with image, get bounding boxes
    res = model.predict(image, conf=CONFIDENCE)
    boxes = res[0].boxes
    # res_plotted = res[0].plot()[:, :, ::-1]
    
    # Convert image to numpy array to crop more easily
    img_array = np.array(image)
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

    # create output directory for bounding box images
    output_dir = ROOT_DIR / 'bboxes'
    output_dir.mkdir(exist_ok=True)
    # delete existing images from before
    for file in output_dir.iterdir():
        if file.is_file():
            file.unlink()
    
    # crop images to square images that contain each bounding box
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
            cropped_pil = PIL.Image.fromarray(cropped)
            
            # get class name if available
            class_id = int(box.cls[0])
            class_name = model.names.get(class_id, f"class_{class_id}")
            confidence_score = box.conf[0]
            
            # Save with descriptive filename
            filename = f"box_{idx:03d}_{class_name}_{confidence_score:.2f}.jpg"
            save_path = output_dir / filename
            
            # Convert RGBA to RGB before saving as JPEG (JPEG doesn't support alpha)
            if cropped_pil.mode == 'RGBA':
                rgb_img = PIL.Image.new('RGB', cropped_pil.size, (255, 255, 255))
                rgb_img.paste(cropped_pil, mask=cropped_pil.split()[3])  # Use alpha as mask
                rgb_img.save(save_path)
            else:
                cropped_pil.save(save_path)
            
            print(f"Saved cropped box {idx}: {filename}")
            print(f"  Original box: [{x1}, {y1}, {x2}, {y2}] (w:{box_width}, h:{box_height})")
            print(f"  Square crop: [{crop_x1}, {crop_y1}, {crop_x2}, {crop_y2}] (size:{square_size}x{square_size})")
            print(f"  Class: {class_name}, Confidence: {confidence_score:.2f}")
            
            # Optional: Draw the original bounding box on the cropped image for verification
            # This helps visualize that the box is centered
            if False:  # Set to True to enable
                from PIL import ImageDraw
                draw = ImageDraw.Draw(cropped_pil)
                # Translate box coordinates to crop coordinates
                box_in_crop = (
                    x1 - crop_x1,
                    y1 - crop_y1,
                    x2 - crop_x1,
                    y2 - crop_y1
                )
                draw.rectangle(box_in_crop, outline=(255, 0, 0), width=2)
                cropped_pil.save(save_path.with_stem(f"{save_path.stem}_with_box"))
    else:
        print("No boxes detected!")

if __name__ == "__main__":
    main()