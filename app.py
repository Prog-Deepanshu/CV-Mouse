import cv2
import mediapipe as mp
import pyautogui
import numpy as np

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

screen_w, screen_h = pyautogui.size()

prev_x, prev_y = screen_w // 2, screen_h // 2
clicking = False
sensitivity_factor = 2.0

def process_frame(frame):
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb_frame)
    return result

def draw_landmarks(frame, landmarks):
    if landmarks:
        for landmark in landmarks:
            mp_drawing.draw_landmarks(frame, landmark, mp_hands.HAND_CONNECTIONS)
    return frame

def smooth_move(x, y, prev_x, prev_y, alpha=0.1):
    smoothed_x = int(alpha * x + (1 - alpha) * prev_x)
    smoothed_y = int(alpha * y + (1 - alpha) * prev_y)
    return smoothed_x, smoothed_y

def perform_actions(landmarks, frame_w, frame_h):
    global prev_x, prev_y, clicking

    if landmarks:
        index_tip = landmarks[0].landmark[8]
        x = int(index_tip.x * frame_w)
        y = int(index_tip.y * frame_h)
        screen_x = screen_w * index_tip.x * sensitivity_factor
        screen_y = screen_h * index_tip.y * sensitivity_factor
        
        screen_x = max(0, min(screen_w - 1, screen_x))
        screen_y = max(0, min(screen_h - 1, screen_y))
        
        screen_x, screen_y = smooth_move(screen_x, screen_y, prev_x, prev_y)
        pyautogui.moveTo(screen_x, screen_y)
        
        prev_x, prev_y = screen_x, screen_y

        thumb_tip = landmarks[0].landmark[4]
        distance = np.sqrt((index_tip.x - thumb_tip.x) ** 2 + (index_tip.y - thumb_tip.y) ** 2)
        
        if distance < 0.05:
            if not clicking:
                pyautogui.mouseDown()
                clicking = True
        else:
            if clicking:
                pyautogui.mouseUp()
                clicking = False

def main():
    cam = cv2.VideoCapture(0)
    cam.set(cv2.CAP_PROP_FPS, 30)

    if not cam.isOpened():
        print("Error: Camera could not be opened.")
        return

    try:
        while True:
            ret, frame = cam.read()
            if not ret:
                print("Error: Frame could not be read.")
                break
            
            frame = cv2.flip(frame, 1)
            frame_h, frame_w, _ = frame.shape
            result = process_frame(frame)

            landmarks = result.multi_hand_landmarks
            if landmarks:
                frame = draw_landmarks(frame, landmarks)
                perform_actions(landmarks, frame_w, frame_h)
            
            cv2.imshow('Hand Controlled Mouse', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        cam.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
