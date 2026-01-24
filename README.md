***NEEDS UPDATE***
***OUT OF DATE VERSION OF README***
Automated Vision Quality Filter 
A lightweight, statistically calibrated computer vision pipeline to filter low-quality frames for robotic vision systems.

# Overview
If a robot attempts to navigate using blurred, pitch-black, or washed-out images, it risks losing localization.This module acts as a pre-processing gatekeeper. It analyzes every frame in real-time and automatically rejects images that do not meet statistical quality standards. Instead of using hard-coded "magic numbers," this system uses a calibration script to learn thresholds from a baseline dataset using Z-score analysis. 

# Key Features 
Motion Blur Detection: Uses Laplacian Variance to measure edge crispness.
Exposure Analysis: Uses Histogram Analysis to detect low-light (underexposure) and glare (overexposure).
Optimized Performance: Implements Short-Circuit Evaluation. Computationally cheap checks (Exposure) run first; expensive checks (Laplacian) only run if necessary, saving battery power.
Statistical Calibration: Includes a dedicated tool (calibrate.py) that scans a "Gold Standard" dataset and calculates robust acceptance thresholds based on the distribution of the data.
# Tech Stack
Python 3.x
OpenCV (cv2): Image processing and color conversion.
NumPy: High-performance array manipulation and boolean masking for histograms.
Git: Version control.
# Project Structure
guidenav_quality_filter/
│
├── data/                       # Dataset storage
│   ├── raw/                    # Incoming frames for testing
│   └── calibration/            # Verified "Good" images for training
│
├── src/                        # Core Logic Module
│   ├── filters.py              # Laplacian & Exposure algorithms
│   └── __init__.py
│
├── calibrate.py                # "Training": Calculates Mean/StdDev thresholds
├── main.py                     # "Inference": The production simulator
└── requirements.txt            # Dependency management
# How It Works
1. Calibration (Offline)We assume "Good" image quality follows a distribution. The calibrate.py script iterates through a verified dataset to calculate: Mean and Standard Deviation for Sharpness, Dark Pixels, and Light Pixels 2. We filter through collected images to decide which to keep and which to discard.
# Setup & Usage 
Install DependenciesBashpython -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
pip install -r requirements.txt
# Run Calibration
Place good images in data/calibration/good/ and run: Bashpython calibrate.py
Output: Returns specific threshold numbers for your camera.
Run FilterUpdate main.py with your new thresholds and run:Bashpython main.py
