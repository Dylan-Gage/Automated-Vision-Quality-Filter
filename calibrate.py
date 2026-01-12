import cv2
import os
import numpy as np
from src.filters import image_blur_score, check_exposure

#image_path = os.path.join("data", "raw" ,"test_image.png") 
#img = cv2.imread(image_path)

folder_path = os.path.join("data", "calibration", "good")

if not os.path.exists(folder_path):
        print(f"Error: folder not found at {folder_path}")
        exit()

file_names = os.listdir(folder_path)
blur_scores = []
dark_ratios = []
light_ratios = []

print(f"Processing {len(file_names)} images...")

for file_name in file_names:
    if not file_name.endswith((".png", ".jpg", ".jpeg")):
         continue
         
    file_path = os.path.join(folder_path, file_name)
    img = cv2.imread(file_path)

    if img is None:
        print(f"Could not load, error at {file_path}")
    else:
        dark_ratio, light_ratio = check_exposure(img)
        b_score = image_blur_score(img)

        dark_ratios.append(dark_ratio)
        light_ratios.append(light_ratio)
        blur_scores.append(b_score)

if len(blur_scores) == 0:
    print("No images processed.")
else:
    b_avg = np.mean(blur_scores)
    b_std = np.std(blur_scores)
    blur_rejection_threshold = b_avg - 1*b_std

    dark_avg = np.mean(dark_ratios)
    dark_std = np.std(dark_ratios)
    dark_rejection_threshold = dark_avg - 1*dark_std

    light_avg = np.mean(light_ratios)
    light_std = np.std(light_ratios)
    light_rejection_threshold = light_avg - 1*light_std

    print("CALIBRATION RESULTS")
    print(f"Blur threshold (min): {blur_rejection_threshold:.2f}") #This number represents the lowest blur score we consider "good" anything below is "bad"
    print(f"Dark threshold (max): {dark_rejection_threshold:.2f}")
    print(f"Light threshold (max): {light_rejection_threshold:.2f}")



#if img is None:
#    print(f"Could not load, error at {image_path}")
#else:
#    score = image_blur_score(img)
#    print(f"Blur score:{score}")
