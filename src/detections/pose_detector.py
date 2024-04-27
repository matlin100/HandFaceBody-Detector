import cv2
import mediapipe as mp

class PoseDetector:
    def __init__(self, static_image_mode=False, model_complexity=1, smooth_landmarks=True, enable_segmentation=False, smooth_segmentation=True, min_detection_confidence=0.5, min_tracking_confidence=0.5):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=static_image_mode,
            model_complexity=model_complexity,
            smooth_landmarks=smooth_landmarks,
            enable_segmentation=enable_segmentation,
            smooth_segmentation=smooth_segmentation,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )
        self.mp_draw = mp.solutions.drawing_utils

    def process_frame(self, image):
        if image is not None and image.size > 0:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = self.pose.process(image_rgb)

            if results.pose_landmarks:
                self.mp_draw.draw_landmarks(
                    image, results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS
                )

            return image
        else:
            print("Invalid frame. Skipping pose detection.")
            return None
