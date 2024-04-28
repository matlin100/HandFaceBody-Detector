import cv2
import mediapipe as mp
import pyautogui
import numpy as np

class HandTracker:
    def __init__(self, min_detection_confidence=0.5, min_tracking_confidence=0.5):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
            max_num_hands=1
        )
        self.mp_draw = mp.solutions.drawing_utils
        self.screen_width, self.screen_height = pyautogui.size()
        self.previous_z = None  # Store the previous z-coordinate of the index finger tip
        self.z_threshold = 0.1  # Threshold for the z-coordinate change to detect a press

    def process_frame(self, image):
        height, width, _ = image.shape
        image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
        results = self.hands.process(image)

        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                self.mp_draw.draw_landmarks(
                    image, hand_landmarks, self.mp_hands.HAND_CONNECTIONS
                )
                index_finger_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_TIP]
                index_x, index_y, index_z = int(index_finger_tip.x * width), int(index_finger_tip.y * height), index_finger_tip.z

                # Convert coordinates to screen resolution
                screen_x = np.interp(index_x, (0, width), (0, self.screen_width))
                screen_y = np.interp(index_y, (0, height), (0, self.screen_height))

                # Detect press movement
                if self.previous_z is not None and self.previous_z - index_z > self.z_threshold:
                    pyautogui.click()  # Perform left-click

                # Update the previous z-coordinate
                self.previous_z = index_z

                # Move the mouse
                pyautogui.moveTo(screen_x, screen_y)

        return image
