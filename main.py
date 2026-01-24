import cv2
import os
import glob
from src.filters import RedundancyFilter, ExposureFilter, SemanticFilter

def main():
    input_folder = "data/raw"
    output_folder = "data/processed_images"
    os.makedirs(output_folder, exist_ok=True)

    redundancy_checker = RedundancyFilter(threshold=50.0)
    exposure_checker = ExposureFilter(sky_crop_ratio=0.3)
    semantic_checker = SemanticFilter(model_size="yolov8n.pt", max_object_coverage=0.10)

    image_files = glob.glob(os.path.join(input_folder, "*.png")) 

    try:
        image_files.sort(key=lambda f: int(''.join(filter(str.isdigit, os.path.basename(f)))))
    except ValueError:
        image_files.sort()

    print(f"Found {len(image_files)} images to process.")

    count_saved = 0

    for image_path in image_files:

        frame = cv2.imread(image_path)

        if frame is None:
            print(f"WARNING: could not read {image_path}")
            continue

        if redundancy_checker.is_duplicate(frame):
            continue

        if not exposure_checker.is_well_exposed(frame):
            continue

        if semantic_checker.has_dynamic_objects(frame):
            continue

        filename = os.path.basename(image_path)
        save_path = os.path.join(output_folder, filename)
        cv2.imwrite(save_path, frame)
        count_saved += 1

        print(f"Saved: {filename}")

    print(f"Processing complete. Saved {count_saved}/{len(image_files)} frames.")

if __name__ == "__main__":
    main()





