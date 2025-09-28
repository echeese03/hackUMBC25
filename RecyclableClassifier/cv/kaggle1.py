import kagglehub
import os
import platform
from pathlib import Path
import cv2
import random
import shutil

# Download the latest version of the dataset
path = kagglehub.dataset_download("isratjahan123/garbage-image-classification")
print("Path to dataset files:", path)

# Parse data into proper format
user_home = Path.home()

# Function to get the dataset path based on OS
def get_kaggle_dataset_path():
    if platform.system() == 'Windows':
        return user_home / '.cache' / 'kagglehub' / 'datasets' / 'isratjahan123' / 'garbage-image-classification' / 'versions' / '1' / 'images' / 'images'
    elif platform.system() == 'Linux':
        return user_home / '.cache' / 'kagglehub' / 'datasets' / 'isratjahan123' / 'garbage-image-classification' / 'versions' / '1' / 'images' / 'images'
    elif platform.system() == 'Darwin':  # macOS
        return user_home / '.cache' / 'kagglehub' / 'datasets' / 'isratjahan123' / 'garbage-image-classification' / 'versions' / '1' / 'images' / 'images'
    else:
        raise OSError("Unsupported Operating System")

# Retrieve the dataset path
dataset_root = get_kaggle_dataset_path()
output_root = user_home / 'organized_dataset'
train_ratio = 0.8

# Dynamically set class names from folder names
class_names = [item for item in os.listdir(dataset_root) if os.path.isdir(os.path.join(dataset_root, item))]
class_dict = {name: idx for idx, name in enumerate(class_names)}
print(class_dict)
print(output_root)

# Create output directories
images_train_dir = output_root / 'images/train'
images_val_dir = output_root / 'images/val'
labels_train_dir = output_root / 'labels/train'
labels_val_dir = output_root / 'labels/val'

for dir in [images_train_dir, images_val_dir, labels_train_dir, labels_val_dir]:
    os.makedirs(dir, exist_ok=True)

# Function to organize the dataset
def organize_dataset():
    for item in os.listdir(dataset_root):
        category_path = os.path.join(dataset_root, item)
        if os.path.isdir(category_path):
            for folder in ['default', 'real_world']:
                folder_path = os.path.join(category_path, folder)
                if os.path.exists(folder_path):
                    for filename in os.listdir(folder_path):
                        if filename.startswith("Image_") and filename.endswith('.png'):
                            image_path = os.path.join(folder_path, filename)

                            # Create a new filename with class tag included
                            base, ext = os.path.splitext(filename)
                            new_filename = f"{item}_{base}{ext}"  # class_name_Image_X.png

                            # Determine whether to move to train or val directory
                            random_val = random.random()
                            split_dir = 'train' if random_val < train_ratio else 'val'
                            dest_image_path = os.path.join(images_train_dir if split_dir == 'train' else images_val_dir, new_filename)

                            # Check for existing files and append a count if necessary
                            count = 1
                            while os.path.exists(dest_image_path):
                                new_filename = f"{item}_{base}_{count}{ext}"  # e.g., class_name_Image_X_1.png
                                dest_image_path = os.path.join(images_train_dir if split_dir == 'train' else images_val_dir, new_filename)
                                count += 1

                            shutil.copy(image_path, dest_image_path)
                            print(f"Moved {filename} to {dest_image_path}")

# Function to draw bounding boxes
class BoxDrawer:
    def __init__(self):
        self.start_point = None
        self.end_point = None
        self.drawing = False
        self.bboxes = []
        self.current_class = None

    def draw_rectangle(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.start_point = (x, y)
            self.drawing = True
        elif event == cv2.EVENT_MOUSEMOVE:
            if self.drawing:
                self.end_point = (x, y)
        elif event == cv2.EVENT_LBUTTONUP:
            self.end_point = (x, y)
            self.drawing = False
            # Add the bounding box with the corresponding class index
            if self.current_class is not None:
                self.bboxes.append((self.start_point[0], self.start_point[1], self.end_point[0], self.end_point[1], self.current_class))

# Function to label images
def label_images(image_path, class_index):
    box_drawer = BoxDrawer()
    box_drawer.current_class = class_index  # Set the default current class based on the folder name
    cv2.namedWindow('Image')
    cv2.setMouseCallback('Image', box_drawer.draw_rectangle)

    while True:
        img = cv2.imread(image_path)
        img_copy = img.copy()

        # Draw the bounding boxes on the image
        for bbox in box_drawer.bboxes:
            cv2.rectangle(img_copy, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)

        # Display the current class being used
        cv2.putText(img_copy, f"Current Class: {class_names[box_drawer.current_class]}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        cv2.imshow('Image', img_copy)
        key = cv2.waitKey(1) & 0xFF

        # Clear existing boxes
        if key == ord('c'):
            box_drawer.bboxes.clear()
            print("Cleared all bounding boxes.")

        # On pressing 'b', save bounding boxes
        if key == ord('b'):
            filename = os.path.basename(image_path)
            base_filename = os.path.splitext(filename)[0]
            unique_filename = f"{base_filename}.txt"

            label_path = os.path.join(labels_train_dir, unique_filename)
            with open(label_path, 'w') as f:
                for bbox in box_drawer.bboxes:
                    x_center = ((bbox[0] + bbox[2]) / 2) / img.shape[1]
                    y_center = ((bbox[1] + bbox[3]) / 2) / img.shape[0]
                    width = (bbox[2] - bbox[0]) / img.shape[1]
                    height = (bbox[3] - bbox[1]) / img.shape[0]
                    f.write(f"{bbox[4]} {x_center} {y_center} {width} {height}\n")
            print(f"Labels saved to {label_path}")
            box_drawer.bboxes.clear()  # Clear drawn boxes after saving

        # On pressing 'q', exit
        if key == ord('q'):
            break

    cv2.destroyAllWindows()

# Main execution
# organize_dataset()  # Call to organize the dataset

# Function to label a specific number of images per class
def label_images_per_class(image_limit):

	for folder in ['train', 'val']:
		folder_path = os.path.join(output_root, f'images/{folder}')
		label_path = os.path.join(output_root, f'labels/{folder}')
		if os.path.exists(folder_path):
			for class_name in class_names:
				class_index = class_dict[class_name]
				labeled_images = 0  # Counter for labeled images for the current class

				# iterate through each image
				for filename in os.listdir(folder_path):
					if filename.endswith('.png'):
						image_path = os.path.join(folder_path, filename)

						# Check if the filename starts with the class name
						if image_path.startswith(os.path.join(folder_path, class_name)):
							# Check if the label file already exists
							label_file = os.path.join(label_path,Path(filename).with_suffix(".txt").name)
							print(f'Checking {label_file}')
							if os.path.exists(label_file):
								print(f"Skipping {image_path}, labels already exist.")
								labeled_images += 1  # Increment count when skipping
							else:
								print(f"Creating label for {image_path} with class {class_name}")     
								label_images(image_path, class_index)  # Call the labeling function
								labeled_images += 1  # Increment the counter

						# Stop if we've labeled the specified number of images for this class
						if labeled_images >= image_limit:
							print(f"Labeled {image_limit} images for class '{class_name}'.")
							break

# Example usage: label only x images per class
label_images_per_class(image_limit=10)

print("Labeling process finished!")