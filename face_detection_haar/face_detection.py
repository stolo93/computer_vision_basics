from contextlib import contextmanager
import cv2 as cv
import argparse
from time import time

from FaceDetector import FaceHaarClassifier


@contextmanager
def VideoCaptureDevice(device: int):
    cap = cv.VideoCapture(device)
    try:
        yield cap
    finally:
        cap.release()


def main():
    parser = argparse.ArgumentParser(
        prog='face_detection',
        description='Face detection using Haar Cascade classifier'
    )
    parser.add_argument('--face', default='data/haarcascade_frontalface_alt.xml')
    parser.add_argument('--eyes', default='data/haarcascade_eye.xml')
    args = parser.parse_args()

    detector = FaceHaarClassifier(
        face_classifier_path=args.face,
        eyes_classifier_path=args.eyes
    )

    previous_time = 0
    with VideoCaptureDevice(0) as cap:
        while True:
            current_time = time()
            fps = int(1 / (current_time - previous_time))
            previous_time = current_time

            _, frame = cap.read()
            detector.detect_and_display(frame)
            cv.putText(frame, f'{fps}', [10, 50], cv.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 3)

            cv.imshow('Face detector', frame)
            if cv.waitKey(1) == ord('q'):
                break


if __name__ == '__main__':
    main()
