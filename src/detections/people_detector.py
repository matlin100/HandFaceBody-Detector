import cv2
import mediapipe as mp

def main():
    video_capture = cv2.VideoCapture(0)  # Using webcam
    people_detector = PeopleDetector()

    while True:
        success, frame = video_capture.read()
        if not success:
            break

        frame = people_detector.detect_people(frame)

        if frame is not None and frame.shape[0] > 0 and frame.shape[1] > 0:
            cv2.imshow('People Detection', frame)
        else:
            print("Invalid frame. Skipping display.")

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video_capture.release()
    cv2.destroyAllWindows()

class PeopleDetector:
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

    def detect_people(self, frame):
        if frame is not None and frame.size > 0:
            results = self.pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            if results.pose_landmarks:
                for i, landmark in enumerate(results.pose_landmarks.landmark):
                    x = int(landmark.x * frame.shape[1])
                    y = int(landmark.y * frame.shape[0])
                    cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)
                    cv2.putText(frame, str(i), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            return frame
        else:
            print("Invalid frame. Skipping people detection.")
            return None

if __name__ == "__main__":
    main()
