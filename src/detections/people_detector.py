import cv2

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
        self.hog = cv2.HOGDescriptor()
        self.hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

    def detect_people(self, frame):
        if frame is not None and frame.size > 0:
            rects, weights = self.hog.detectMultiScale(frame, winStride=(4, 4), padding=(8, 8), scale=1.05)
            for i, (x, y, w, h) in enumerate(rects):
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(frame, f"Person {i+1}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            return frame
        else:
            print("Invalid frame. Skipping people detection.")
            return None

if __name__ == "__main__":
    main()
