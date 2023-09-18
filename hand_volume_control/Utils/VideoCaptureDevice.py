import cv2
from contextlib import contextmanager


@contextmanager
def VideoCaptureDevice(device: int = 0):
    cap = cv2.VideoCapture(device)
    try:
        yield cap
    finally:
        cap.release()


def main():
    with VideoCaptureDevice(0) as cap:
        while True:
            _, frame = cap.read()
            cv2.imshow('Frame', frame)
            if cv2.waitKey(1) == ord('q'):
                break


if __name__ == '__main__':
    main()