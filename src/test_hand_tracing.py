import cv2
from src.detections.detector_init import initialize_hand_tracker

def main():
    video_capture = cv2.VideoCapture(0)  # Using webcam

    hand_tracker = initialize_hand_tracker()
    while True:
        ret, frame = video_capture.read()
        if not ret:
            break

        frame = hand_tracker.process_frame(frame)
        frame = frame[0] if isinstance(frame, tuple) else frame

        cv2.imshow('Multi-Object Detection', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video_capture.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
