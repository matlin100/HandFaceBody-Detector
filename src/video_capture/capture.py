import cv2

def capture_video(video_path=None):
    """
    Initialize video capture.
    If a video_path is provided, open the video file. Otherwise, open the default camera.
    """
    if video_path:
        cap = cv2.VideoCapture(video_path)  # Open video file
    else:
        cap = cv2.VideoCapture(0)  # Open default camera
    return cap

def show_frame(frame, window_name='Video'):
    cv2.imshow(window_name, frame)

def release_video(capture):
    capture.release()

def destroy_windows():
    cv2.destroyAllWindows()
