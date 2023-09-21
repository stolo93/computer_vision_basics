import cv2
import mediapipe as mp
import time


class HandDetector:
    def __init__(self,
                 mode=False,
                 max_hands=2,
                 min_detection_confidence=0.5,
                 min_tracking_confidence=0.5):
        self.hands = mp.solutions.hands.Hands(
            static_image_mode=mode,
            max_num_hands=max_hands,
            min_tracking_confidence=min_tracking_confidence,
            min_detection_confidence=min_detection_confidence
        )
        self.mp_draw = mp.solutions.drawing_utils

    def find_hands(self, img, draw: bool):
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.hands.process(img_rgb)
        if results.multi_hand_landmarks is not None:
            for hand_landmarks in results.multi_hand_landmarks:
                if draw:
                    self.mp_draw.draw_landmarks(img, hand_landmarks, mp.solutions.hands.HAND_CONNECTIONS)
        return img

    def find_position(self, img, hand_number=0, draw=True):
        lm_list = []
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.hands.process(img_rgb)

        if results.multi_hand_landmarks:
            height, width, channels = img.shape
            my_hand = results.multi_hand_landmarks[hand_number]
            for landmark_id, landmark in enumerate(my_hand.landmark):
                cx, cy = int(landmark.x * width), int(landmark.y * height)
                lm_list.append([landmark_id, cx, cy])
                if draw:
                    cv2.circle(img, (cx, cy), 5, (255, 0, 0), cv2.FILLED)
        return lm_list


def main():
    current_time = 0
    previous_time = 0

    cap = cv2.VideoCapture(0)
    detector = HandDetector()
    finger_tips = {
        'thumb': 4,
        'index_finger': 8,
        'middle_finger': 12,
        'ring_finger': 16,
        'pinky': 20
    }

    while True:
        _, frame = cap.read()
        landmarks = detector.find_position(frame)
        if len(landmarks) != 0:
            middle_finger_tip = landmarks[finger_tips['middle_finger']]
            cv2.circle(frame, (middle_finger_tip[1], middle_finger_tip[2]), 10, (0, 255, 0), cv2.FILLED)
        current_time = time.time()
        fps = int(1/(current_time - previous_time))
        previous_time = current_time

        cv2.putText(frame, str(fps), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(255, 0, 255), thickness=3)
        cv2.imshow('Frame', frame)
        if cv2.waitKey(1) == ord('q'):
            break

    cap.release()


if __name__ == "__main__":
    main()