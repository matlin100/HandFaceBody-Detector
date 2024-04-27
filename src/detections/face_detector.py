import cv2

class FaceDetector:
    def __init__(self):
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.trackers = {}
        self.face_times = {}
        self.last_update_time = {}
        self.face_id_counter = 0

    def detect_and_track_faces(self, frame, current_time):
        if current_time - self.last_update_time.get('detect', 0) > 5:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30), flags=cv2.CASCADE_SCALE_IMAGE)
            detected_faces = []

            for (x, y, w, h) in faces:
                detected_faces.append((x, y, w, h))

            for face_rect in detected_faces:
                x, y, w, h = face_rect
                match_found = False

                for tracker_id, tracker in list(self.trackers.items()):
                    ok, box = tracker.update(frame)
                    if ok:
                        bx, by, bw, bh = [int(v) for v in box]
                        # Check for overlap between existing tracker box and new detected box
                        if self.do_boxes_overlap((x, y, w, h), (bx, by, bw, bh)):
                            match_found = True
                            break

                if not match_found:
                    tracker = cv2.TrackerCSRT_create()
                    tracker_id = self.face_id_counter
                    self.trackers[tracker_id] = tracker
                    tracker.init(frame, (x, y, w, h))
                    self.face_times[tracker_id] = {'start': current_time, 'duration': 0}
                    self.face_id_counter += 1

            self.last_update_time['detect'] = current_time

        # Update and draw trackers
        for tracker_id, tracker in list(self.trackers.items()):
            ok, box = tracker.update(frame)
            if ok:
                x, y, w, h = [int(v) for v in box]
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                duration = int(current_time - self.face_times[tracker_id]['start'])
                self.face_times[tracker_id]['duration'] = duration
                cv2.putText(frame, f"Face {tracker_id}: {duration}s", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        return frame

    def do_boxes_overlap(self, box1, box2):
        x1, y1, w1, h1 = box1
        x2, y2, w2, h2 = box2
        return not (x1 + w1 < x2 or x2 + w2 < x1 or y1 + h1 < y2 or y2 + h2 < y1)
