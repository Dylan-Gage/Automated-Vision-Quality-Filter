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

