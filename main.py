import cv2
import os
from src.filters import is_image_acceptable

BLUR_THRESHOLD = 24.61 #Should be your real threshold

folder_path = os.path.join("data", "raw")

if not os.path.exists(folder_path):
        print(f"Error: folder not found at {folder_path}")
        exit()

file_names = os.listdir(folder_path)
acceptable_images = []

print(f"Processing {len(file_names)} images...")

for file_name in file_names:
    if not file_name.endswith((".png", ".jpg", ".jpeg")):
         continue
         
    file_path = os.path.join(folder_path, file_name)
    img = cv2.imread(file_path)

    if img is None:
        print(f"Error: Could not load image at {file_path}")
    else:
        if is_image_acceptable(img, BLUR_THRESHOLD):
            acceptable_images.append(file_name)

print(acceptable_images)

