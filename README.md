# SLAM Pre-processing Pipeline: Good Frame Extraction (V2.0)
## Overview
This pipeline acts as a Gatekeeper for raw robot camera data. By filtering out "bad" frames before they reach the SLAM algorithm, we maximize map quality and reduce computational waste.
The pipeline follows a Fail-Fast architecture, where filters are ordered from lowest to highest computational cost.

## Pipeline Stages 
1. Redundancy: Rejects identical frames (e.g., robot is docked/stopped). Uses a 64x64 downsampled thumbnail to ignore sensor noise.
2. Movement: Differentiates camera motion from scene motion. Uses Median Flow of ORB features to ignore outliers like reflections or passing pedestrians.
3. Exposure: Slices the image to ignore the sky (top 30%). It ensures the ground/walls (the features SLAM needs) are well-exposed regardless of sky glare.
4. Blur: Uses Laplacian Variance to ensure only sharp frames are used for feature tracking.
5. Semantic: The most expensive step. Uses YOLOv8 to detect dynamic objects (people, cars). Rejects frames if these objects occupy too much of the view.

## Memory & Performance Optimizations
-Vectorization: Feature shift calculations in the MovementFilter are vectorized using NumPy to avoid Python loop bottlenecks. <br>
-Zero-Copy ROI: The ExposureFilter uses NumPy Views rather than Copies when slicing the ground ROI to save memory bandwidth. <br>
-Broadcasting Safety: Grayscale conversion is enforced before MSE calculations to prevent (H,W,C) vs (H,W) subtraction crashes. <br>

## Setup
1. Requirements: Python 3.8+, opencv-python, numpy, and ultralytics.
2. Data Structure: * Place raw .png files in data/raw/. <br>
   -Run python main.py. <br>
   -Cleaned frames will appear in data/processed_images/.

## Sorting Logic
The pipeline uses a custom Lambda Sort Key for file ingestion. This ensures that 10.png correctly follows 2.png (numerical order) rather than following 1.png (alphabetical order), which is critical for maintaining the temporal consistency required by SLAM.
