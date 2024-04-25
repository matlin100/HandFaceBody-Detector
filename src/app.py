import cv2
from hand_tracking import tracker
from video_capture import capture
from people_detection.detector import PeopleDetector  # Directly import the class
from face_detection.detector import FaceDetector  # Directly import the class

def main():
    video_path = '/Users/yechezkelmatlin/PycharmProjects/track me/Screen Recording 2024-04-25 at 12.29.32.mov'
    video_capture = capture.capture_video(video_path or 0)
    hand_tracker = tracker.HandTracker()
    people_detector = PeopleDetector()  # Use the directly imported class
    face_detector = FaceDetector()  # Use the directly imported class

    while video_capture.isOpened():
        success, image = video_capture.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue

        image = hand_tracker.process_frame(image)
        image = people_detector.detect_people(image)
        image = face_detector.detect_faces(image)
        capture.show_frame(image, 'MediaPipe Hands, People and Face Detection')

        if cv2.waitKey(5) & 0xFF == 27:  # Press 'Esc' to exit
            break

    capture.release_video(video_capture)
    capture.destroy_windows()

if __name__ == "__main__":
    main()