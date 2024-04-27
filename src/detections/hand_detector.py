import cv2
import mediapipe as mp
import pyautogui
import numpy as np  # Ensure NumPy is imported

class HandTracker:
    def __init__(self, min_detection_confidence=0.5, min_tracking_confidence=0.5):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )
        self.mp_draw = mp.solutions.drawing_utils
        self.screen_width, self.screen_height = pyautogui.size()  # Get the size of the screen

    def process_frame(self, image):
        height, width, _ = image.shape
        image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
        results = self.hands.process(image)

        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        pinch_detected = False
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                self.mp_draw.draw_landmarks(
                    image, hand_landmarks, self.mp_hands.HAND_CONNECTIONS
                )
                thumb_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.THUMB_TIP]
                index_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_TIP]

                # Calculate distance for pinch gesture
                distance = self.calculate_distance(thumb_tip, index_tip)
                if distance < 0.05:  # Threshold for pinch gesture
                    pinch_detected = True
                    pyautogui.click()  # Perform left-click

                # Get coordinates of index finger tip for moving mouse
                index_finger_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_TIP]
                index_x, index_y = int(index_finger_tip.x * width), int(index_finger_tip.y * height)

                # Convert coordinates to screen resolution
                screen_x = np.interp(index_x, (0, width), (0, self.screen_width))
                screen_y = np.interp(index_y, (0, height), (0, self.screen_height))

                # Move the mouse
                pyautogui.moveTo(screen_x, screen_y)

        return image, pinch_detected

    def calculate_distance(self, point1, point2):
        return ((point1.x - point2.x) ** 2 + (point1.y - point2.y) ** 2) ** 0.5
