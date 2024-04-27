import cv2
import mediapipe as mp

class FaceDetector:
    def __init__(self):
        self.mp_face_detection = mp.solutions.face_detection
        self.face_detection = self.mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5)
        self.trackers = {}
        self.face_times = {}
        self.last_update_time = {}
        self.face_id_counter = 0

    def detect_and_track_faces(self, frame, current_time):
        if current_time - self.last_update_time.get('detect', 0) > 5:
            results = self.face_detection.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            detected_faces = []

            if results.detections:
                for detection in results.detections:
                    bounding_box = detection.location_data.relative_bounding_box
                    x = int(bounding_box.xmin * frame.shape[1])
                    y = int(bounding_box.ymin * frame.shape[0])
                    w = int(bounding_box.width * frame.shape[1])
                    h = int(bounding_box.height * frame.shape[0])
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
                if x >= 0 and y >= 0 and x + w < frame.shape[1] and y + h < frame.shape[0]:
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    duration = int(current_time - self.face_times[tracker_id]['start'])
                    self.face_times[tracker_id]['duration'] = duration
                    cv2.putText(frame, f"Face {tracker_id}: {duration}s", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                else:
                    # Remove the tracker if the face goes out of the frame
                    del self.trackers[tracker_id]
                    del self.face_times[tracker_id]

        return frame

    def do_boxes_overlap(self, box1, box2):
        x1, y1, w1, h1 = box1
        x2, y2, w2, h2 = box2
        return not (x1 + w1 < x2 or x2 + w2 < x1 or y1 + h1 < y2 or y2 + h2 < y1)
