import cv2
import numpy as np
from ultralytics import YOLO

class RedundancyFilter():
    """Rejects identical or near-identical frames to reduce dataset bloat."""

    def __init__(self, threshold=50.0, thumb_size=(64, 64)):
        """
        Args:
            threshold (float): MSE value below which a frame is considered a duplicate.
            thumb_size (tuple): Downscale size for noise-invariant comparison.
        """
        self.threshold = threshold
        self.thumb_size = thumb_size
        self.last_thumb_frame = None

    def reset(self):
        """Clears state for new video sequences."""
        self.last_thumb_frame = None

    def is_duplicate(self, frame):
        """
        Uses Perceptual Hashing (MSE) to detect static robot states.
        
        Why: Downscaling to 64x64 via INTER_AREA acts as a low-pass filter, 
        removing sensor noise that would otherwise trigger a false 'move' detection.
        """
        # Downscale and grayscale to ensure (H, W) matches for subtraction
        thumb = cv2.resize(frame, self.thumb_size, interpolation=cv2.INTER_AREA)
        thumb = cv2.cvtColor(thumb, cv2.COLOR_BGR2GRAY)

        if self.last_thumb_frame is None:
            self.last_thumb_frame = thumb
            return False
        
        # Calculate Mean Squared Error
        err = np.sum((thumb.astype("float") - self.last_thumb_frame.astype("float")) ** 2)
        err /= float(thumb.shape[0] * thumb.shape[1])

        if err < self.threshold:
            return True
        else:
            # Only update reference if we actually moved significantly
            self.last_thumb_frame = thumb
            return False

class MovementFilter():
    """Differentiates camera motion from scene motion (like reflections)."""

    def __init__(self, min_translation):
        """
        Args:
            min_translation (float): Minimum median pixel shift to count as movement.
        """
        self.min_translation = min_translation
        self.last_keypoints = None
        self.last_descriptors = None

        self.orb = cv2.ORB_create(nfeatures=500)
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    def is_moving(self, frame):
        """
        Calculates global camera movement using the median shift of ORB features.
        
        Why: Using the Median rather than Mean allows us to ignore 'outlier' 
        motion, such as a person walking by or a reflection in a glass door 
        while the robot is stationary.
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        keypoints, descriptors = self.orb.detectAndCompute(gray, None)

        if self.last_keypoints is None:
            self.last_keypoints = keypoints
            self.last_descriptors = descriptors
            return True
        if descriptors is None or len(descriptors) < 10:
            return True
        
        matches = self.matcher.match(self.last_descriptors, descriptors)

        matches = sorted(matches, key=lambda x: x.distance)
        good_matches = matches[:int(len(matches) * 0.5)]

        if len(good_matches) < 5:
            self.last_keypoints = keypoints
            self.last_descriptors = descriptors
            return True
        
        pts_prev = np.array([self.last_keypoints[m.queryIdx].pt for m in good_matches])
        pts_curr = np.array([keypoints[m.trainIdx].pt for m in good_matches])

        distances = np.linalg.norm(pts_prev - pts_curr, axis=1)
        median_shift = np.median(distances)

        # Calculate Euclidean distances for all points at once
        if median_shift < self.min_translation:
            return False
        
        self.last_descriptors, self.last_keypoints = descriptors, keypoints
        return True

class ExposureFilter():
    """Rejects frames that are under or over-exposed in the ROI."""
    def __init__(self, sky_exclusion_ratio=0.3, min_threshold=40, max_threshold=220):
        self.sky_exclusion_ratio = sky_exclusion_ratio
        self.min_threshold = min_threshold
        self.max_threshold = max_threshold

        if min_threshold >= max_threshold:
            raise ValueError("min_threshold must be less than max_threshold")
    
    def is_well_exposed(self, frame):
        """
        Analyzes brightness only on the ground/ROI.
        
        Why: SLAM relies on floor/wall features. A bright sky can cause 
        auto-exposure to underexpose the ground, making features undiscernible. 
        We slice the frame to focus the 'Gatekeeper' on relevant data.
        """
        h, w = frame.shape[:2]

        crop_start = int(h*self.sky_exclusion_ratio)

        ground_roi = frame[crop_start:, :]
        gray_roi = cv2.cvtColor(ground_roi, cv2.COLOR_BGR2GRAY)

        avg_brightness = np.mean(gray_roi)

        return self.min_threshold <= avg_brightness <= self.max_threshold
    

class SemanticFilter():
    """Prevents dynamic objects from corrupting the static map."""

    def __init__(self, model_size="yolov8n.pt",max_object_coverage=0.05):
        self.model = YOLO(model_size)
        self.dynamic_classes = [0,1,2,3,5,7] # Person, bicycle, car, motorcycle, bus, truck
        self.max_ratio = max_object_coverage

    def has_dynamic_objects(self, frame):
        """
        Calculates the ratio of the frame occupied by moving COCO classes.
        
        Why: Small dynamic objects in the distance don't hurt SLAM much, but 
        large objects (like a person standing in front of the robot) break 
        the static world assumption.
        """
        results = self.model(frame, verbose=False)
        dynamic_area = 0.0

        for r in results:
            for box in r.boxes:
                if int(box.cls[0]) in self.dynamic_classes:
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    dynamic_area += (x2 - x1) * (y2 - y1)
            
        ratio = dynamic_area / (frame.shape[0] * frame.shape[1])
        return ratio > self.max_ratio

class BlurFilter():
    def __init__(self, blur_threshold=100.0):
        self.blur_threshold = blur_threshold

    def image_blur_score(self, frame):
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        variance = cv2.Laplacian(gray, cv2.CV_64F).var()
        return variance > self.blur_threshold


