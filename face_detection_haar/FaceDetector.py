"""
Face and eyes detector module
"""

import cv2 as cv
import numpy as np


class FaceHaarClassifier:
    def __init__(self,
                 face_classifier_path: str,
                 eyes_classifier_path: str
                 ):
        self.face_classifier = cv.CascadeClassifier(face_classifier_path)
        self.eyes_classifier = cv.CascadeClassifier(eyes_classifier_path)
        assert self.face_classifier is not None, 'Face classifier not initialized'

    def detect_and_display(self, frame: np.ndarray):
        frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        frame_gray = cv.equalizeHist(frame_gray)

        # Detect faces
        faces = self.face_classifier.detectMultiScale(
            image=frame_gray,
            minNeighbors=4,
            scaleFactor=1.05,
            minSize=(30, 30)
        )

        for x, y, w, h in faces:
            center = (x + w//2), (y + h//2)
            cv.ellipse(frame, center, (w//2, h//2), 0, 0, 360, (255, 0, 255), 4)

            if self.eyes_classifier is None:
                continue
            face_roi = frame_gray[y:y+h, x:x+w]
            eyes = self.eyes_classifier.detectMultiScale(face_roi)

            for xi, yi, wi, hi in eyes:
                center_i = x+xi+wi//2, y+yi+hi//2
                cv.circle(frame, center_i, int(wi//2), (0, 200, 0), 3)