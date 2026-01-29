import cv2
import os
import glob
from src.filters import RedundancyFilter, ExposureFilter, SemanticFilter, BlurFilter, MovementFilter

def main():
    """
    Main execution loop for the Good Frame Extraction pipeline.
    
    This script processes raw robot camera feeds through a series of 'Gatekeeper' 
    filters to ensure only high-quality, static, and well-exposed frames are 
    passed to the SLAM algorithm.
    """
    input_folder = "data/raw"
    output_folder = "data/processed_images"
    os.makedirs(output_folder, exist_ok=True)

    # Initialize filters with thresholds
    redundancy_checker = RedundancyFilter(threshold=99.0)
    exposure_checker = ExposureFilter(sky_exclusion_ratio=0.3)
    semantic_checker = SemanticFilter(model_size="yolov8n.pt", max_object_coverage=0.10)
    blur_checker = BlurFilter(blur_threshold=500.0)
    movement_checker = MovementFilter(min_translation=2.0)

    # Gather image files
    image_files = glob.glob(os.path.join(input_folder, "*.png")) 

    # Why: SLAM requires chronological order. The lambda sorts '10.png' after '2.png'
    # instead of lexicographical order ('10.png' before '2.png').
    try:
        image_files.sort(key=lambda f: int(''.join(filter(str.isdigit, os.path.basename(f)))))
    except ValueError:
        image_files.sort()

    print(f"Found {len(image_files)} images to process.")

    count_saved = 0

    # Pipeline Execution: Ordered from Lowest Computational Cost to Highest
    for image_path in image_files:

        frame = cv2.imread(image_path)

        if frame is None:
            print(f"WARNING: could not read {image_path}")
            continue

        # 1. Redundancy (Lowest Cost: Simple MSE on 64x64 thumbnail)
        if redundancy_checker.is_duplicate(frame):
            continue

        # 2. Movement (Medium Cost: ORB Feature Matching)
        if not movement_checker.is_moving(frame):
            continue

        # 3. Exposure (Medium Cost: ROI Mean Brightness)
        if not exposure_checker.is_well_exposed(frame):
            continue

        # 4. Blur (Medium Cost: Laplacian Variance)
        if not blur_checker.image_blur_score(frame):
            continue

        # 5. Semantic (Highest Cost: YOLOv8 Inference)
        if semantic_checker.has_dynamic_objects(frame):
            continue

        # All gates passed: Save the 'Good Frame'
        filename = os.path.basename(image_path)
        save_path = os.path.join(output_folder, filename)
        cv2.imwrite(save_path, frame)
        count_saved += 1

        #print(f"Saved: {filename}")

    print(f"Processing complete. Saved {count_saved}/{len(image_files)} frames.")

if __name__ == "__main__":
    main()





