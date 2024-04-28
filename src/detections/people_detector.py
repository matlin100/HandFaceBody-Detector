import cv2

class PeopleDetector:
    def __init__(self):
        # Initialize the HOG descriptor/person detector
        self.hog = cv2.HOGDescriptor()
        self.hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

    def detect_people(self, frame):
        # Check if the frame is valid
        if frame is not None and frame.size > 0:
            # Convert frame to grayscale
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Perform people detection
            rects, weights = self.hog.detectMultiScale(gray, winStride=(4, 4), padding=(8, 8), scale=1.05)

            # Draw bounding boxes and labels on the original frame
            for i, (x, y, w, h) in enumerate(rects):
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(frame, f"Person {i+1}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            return frame
        else:
            print("Invalid frame. Skipping people detection.")
            return None
