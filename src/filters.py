import cv2
import numpy as np

def image_blur_score(image):
    """
    Calculates the variance of the Laplacian to detect blur
    High variance = Sharper Image
    Low variance = Blurry Image
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    variance = cv2.Laplacian(gray, cv2.CV_64F).var()
    return variance


def check_exposure(image):
    """
    Calculates the percentage of pixels that are too dark or too bright.
    Returns: (dark_ratio, light_ratio)
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    total_pixels = gray.shape[0] * gray.shape[1]

    num_dark_pixels = np.sum(gray < 30)
    num_light_pixels = np.sum(gray > 225)

    dark_ratio = num_dark_pixels / total_pixels
    light_ratio = num_light_pixels / total_pixels

    return dark_ratio, light_ratio
    

def is_image_acceptable(image, blur_threshold, dark_threshold=0.5, light_threshold=0.1):
    """
    Master Gatekeeper Function.
    1. Checks Exposure (Fast)
    2. Checks Blur (Slow)
    Returns: True (Keep) or False (Reject)
    """
    dark_ratio, light_ratio = check_exposure(image)
    if dark_ratio > dark_threshold or light_ratio > light_threshold:
        return False
    
    blur_score = image_blur_score(image)
    return blur_score > blur_threshold
