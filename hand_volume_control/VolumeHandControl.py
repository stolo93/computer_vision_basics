import cv2
import time
from math import hypot

from VolumeControl import set_master_volume
from HandTrackingModule import HandDetector

import sys
sys.path.append('..')
from Utils.VideoCaptureDevice import VideoCaptureDevice


def translate(value, leftMin, leftMax, rightMin, rightMax):
    # Figure out how 'wide' each range is
    leftSpan = leftMax - leftMin
    rightSpan = rightMax - rightMin

    # Convert the left range into a 0-1 range (float)
    valueScaled = float(value - leftMin) / float(leftSpan)

    # Convert the 0-1 range into a value in the right range.
    return rightMin + (valueScaled * rightSpan)


def main():
    show_cam: bool = True if len(sys.argv) > 1 and sys.argv[1] == '--cam' else False

    CAM_HEIGHT, CAM_WIDTH = 640, 460
    FINGER_TIPS = {
        'thumb': 4,
        'index': 8
    }

    current_time = 0
    previous_time = 0

    detector = HandDetector(min_detection_confidence=0.8)

    with VideoCaptureDevice(0) as cap:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_HEIGHT)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_WIDTH)
        while True:
            success, img = cap.read()
            landmarks = detector.find_position(img, draw=False)

            if len(landmarks) != 0:
                thumb_tip = landmarks[FINGER_TIPS['thumb']]
                index_finger_tip = landmarks[FINGER_TIPS['index']]

                x1, y1 = thumb_tip[1], thumb_tip[2]
                x2, y2 = index_finger_tip[1], index_finger_tip[2]
                if show_cam:
                    cv2.circle(img, (x1, y1), 10, (255, 255, 0), cv2.FILLED)
                    cv2.circle(img, (x2, y2), 10, (255, 255, 0), cv2.FILLED)
                    cv2.line(img, (x1, y1), (x2, y2), (255, 0, 255), 3)

                    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                    cv2.circle(img, (cx, cy), 10, (255, 0, 255), cv2.FILLED)

                length = hypot(x2 - x1, y2 - y1)
                length = 150 if length > 150 else length

                volume = translate(length, 30, 150, 0, 100)
                volume = 0 if volume < 0 else volume
                set_master_volume(volume)

            current_time = time.time()
            fps = 1 / (current_time - previous_time)
            previous_time = current_time

            if show_cam:
                cv2.putText(img, f'FPS: {int(fps)}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), thickness=2)
                cv2.imshow('Frame', img)
                if cv2.waitKey(1) == ord('q'):
                    break


if __name__ == '__main__':
    main()