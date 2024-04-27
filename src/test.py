import cv2
import time
from src.detections.detector_init import initialize_face_detector, initialize_hand_tracker, initialize_people_detector, initialize_pose_detector

def main():
    video_capture = cv2.VideoCapture(0)  # Using webcam
    # Initialize detectors using functions from detector_init.py
    face_detector = initialize_face_detector()
    hand_tracker = initialize_hand_tracker()
    people_detector = initialize_people_detector()
    pose_detector = initialize_pose_detector()

    # Initialize the start time right before the loop starts
    start_time = time.time()

    while True:
        ret, frame = video_capture.read()
        if not ret:
            break

        # Use the correct reference to start_time
        current_time = int(time.time() - start_time)

        # Process detections
        if face_detector:
            frame = face_detector.detect_and_track_faces(frame, current_time)
        if hand_tracker:
            frame = hand_tracker.process_frame(frame)
        if people_detector:
            image = frame[0] if isinstance(frame, tuple) else frame
            frame = people_detector.detect_people(image)
        if pose_detector:
            image = frame[0] if isinstance(frame, tuple) else frame
            frame = pose_detector.process_frame(image)

        # Display only if frame is valid
        if frame is not None and frame.shape[0] > 0 and frame.shape[1] > 0:
            cv2.imshow('Multi-Object Detection', frame)
        else:
            print("Invalid frame received. Skipping display.")

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video_capture.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
