import cv2
import numpy as np
from ultralytics import YOLO

class RedundancyFilter():
    def __init__(self, threshold=50.0, thumb_size=(64, 64)):
        self.threshold = threshold
        self.thumb_size = thumb_size
        self.last_thumb_frame = None

    def is_duplicate(self, frame):
        thumb = cv2.resize(frame, self.thumb_size, interpolation=cv2.INTER_AREA)
        thumb = cv2.cvtColor(thumb, cv2.COLOR_BGR2GRAY)

        if self.last_thumb_frame is None:
            self.last_thumb_frame = thumb
            return False
        
        err = np.sum((thumb.astype("float") - self.last_thumb_frame.astype("float")) ** 2)
        err /= float(thumb.shape[0] * thumb.shape[1])

        if err < self.threshold:
            return True
        else:
            self.last_thumb_frame = thumb
            return False

class ExposureFilter():
    def __init__(self, sky_crop_ratio=0.3, min_threshold=40, max_threshold=220):
        self.sky_crop_ratio = sky_crop_ratio
        self.min_threshold = min_threshold
        self.max_threshold = max_threshold
    
    def is_well_exposed(self, frame):
        h, w = frame.shape[:2]

        crop_start = int(h*self.sky_crop_ratio)

        ground_roi = frame[crop_start:, :]
        gray_roi = cv2.cvtColor(ground_roi, cv2.COLOR_BGR2GRAY)

        avg_brightness = np.mean(gray_roi)

        if avg_brightness < self.min_threshold:
            return False
        elif avg_brightness > self.max_threshold:
            return False
        return True
    

class SemanticFilter():
    def __init__(self, model_size="yolov8n.pt",max_object_coverage=0.05):
        self.model = YOLO(model_size)
        self.dynamic_classes = [0,1,2,3,5,7]
        self.max_ratio = max_object_coverage

    def has_dynamic_objects(self, frame):
        h, w = frame.shape[:2]
        total_pixels = h*w

        results = self.model(frame, verbose=False)
        dynamic_pixel_count = 0

        for r in results:
            boxes = r.boxes
            for box in boxes:
                cls_id = int(box.cls[0])
                if cls_id in self.dynamic_classes:
                    x1, y1, x2, y2 = box.xyxy[0]
                    box_area = (x2-x1) * (y2-y1)
                    dynamic_pixel_count += box_area
            
        ratio = dynamic_pixel_count / total_pixels
        if ratio > self.max_ratio:
            return True
        return False

class BlurFilter():
    def __init__(self, blur_threshold=100.0):
        self.blur_threshold = blur_threshold

    def image_blur_score(self, frame):
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        variance = cv2.Laplacian(gray, cv2.CV_64F).var()
        return variance > self.blur_threshold


