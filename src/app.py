import cv2
import time
from src.detections.detector_init import initialize_face_detector, initialize_hand_tracker, initialize_people_detector, initialize_pose_detector

def main():
    video_capture = cv2.VideoCapture(0)  # Using webcam

    # Initialize detectors using functions from detector_init.py
    face_detector = initialize_face_detector()
    hand_tracker = initialize_hand_tracker()
    pose_detector = initialize_pose_detector()
    start_time = time.time()

    while True:
        ret, frame = video_capture.read()
        if not ret:
            print("Failed to capture video frame.")
            break

        # Hand tracking
        result = hand_tracker.process_frame(frame) if hand_tracker is not None else (frame,)

        # Assuming result is a tuple and the frame is the first element
        processed_frame = result[0] if result and isinstance(result, tuple) else None

        # Display only if frame is valid
        if processed_frame is not None and processed_frame.shape[0] > 0 and processed_frame.shape[1] > 0:
            cv2.imshow('Multi-Object Detection', processed_frame)
        else:
            print("Invalid frame received. Skipping display.")

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Terminating program.")
            break

    video_capture.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
