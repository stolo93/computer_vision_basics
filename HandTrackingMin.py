import cv2
import mediapipe as mp
import time


cap = cv2.VideoCapture(0)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
mp_draw = mp.solutions.drawing_utils

previous_time = 0
current_time = 0

while True:
    success, frame = cap.read()
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    if results.multi_hand_landmarks is not None:
        for hand_landmarks in results.multi_hand_landmarks:
            for landmark_id, landmark in enumerate(hand_landmarks.landmark):
                height, width, channels = frame.shape
                cx, cy = int(landmark.x * width), int(landmark.y * height)
                
            mp_draw.draw_landmarks(frame, hand_landmarks, connections=mp_hands.HAND_CONNECTIONS)

    current_time = time.time()
    fps = int(1/(current_time - previous_time))
    previous_time = current_time

    cv2.putText(frame, str(fps), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(255, 0, 255), thickness=3)

    cv2.imshow('Image', frame)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()