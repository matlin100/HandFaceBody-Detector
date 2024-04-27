from .face_detector import FaceDetector
from .hand_detector import HandTracker
from .people_detector import PeopleDetector
from .pose_detector import PoseDetector

def initialize_face_detector():
    try:
        return FaceDetector()
    except Exception as e:
        print(f"Error initializing FaceDetector: {e}")
        return None

def initialize_hand_tracker():
    try:
        return HandTracker()
    except Exception as e:
        print(f"Error initializing HandTracker: {e}")
        return None

def initialize_people_detector():
    try:
        return PeopleDetector()
    except Exception as e:
        print(f"Error initializing PeopleDetector: {e}")
        return None

def initialize_pose_detector():
    try:
        return PoseDetector()
    except Exception as e:
        print(f"Error initializing PoseDetector: {e}")
        return None
