import cv2
import numpy as np

class RedundancyFilter():
    def __init__(self, threshold=50.0, thumb_size=(64, 64)):
        self.threshold = threshold
        self.thumb_size = thumb_size
        self.last_thumb_frame = None

    def is_duplicate(self, frame):
        thumb = cv2.resize(frame, self.thumb_size, interpolation=cv2.INTER_AREA)
        thumb = cv2.color(frame, cv2.COLOR_BGR2GRAY)

        if self.last_thumb_frame is None:
            self.last_thumb_frame = frame
            return False
        
        err = np.sum((thumb.astype("float") - self.last_thumb_frame.astype("float")) ** 2)
        err /= float(thumb.shape[0] * thumb.shape[1])

        if err < self.threshold:
            return True
        else:
            self.last_thumb_frame = frame
            return False

class ExposureFilter():
    def __init__(self, sky_crop_ratio=0.3, min_threshold=40, max_threshold=220):
        self.sky_crop_ratio = sky_crop_ratio
        self.min_threshold = min_threshold
        self.max_threshold = max_threshold
    
    def is_well_exposed(self, frame):
        h,w = frame.shape[:2]

        crop_start = int(h*self.sky_crop_ratio)

        ground_roi = frame[crop_start:, :]
        gray_roi = cv2.cvtColor(ground_roi, cv2.COLOR_BGR2GRAY)

        avg_brightness = np.mean(gray_roi)

        if avg_brightness < self.min_threshold:
            return False
        elif avg_brightness > self.max_threshold:
            return False
        return True
    





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
