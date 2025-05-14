import time
import dlib
import cv2
import numpy as np
import os
import csv
from collections import deque
from datetime import datetime
from imutils import face_utils
from imutils.video import FileVideoStream, VideoStream

class BlinkDetector:
    def __init__(self, shape_predictor_path, eye_ar_thresh=0.18, consec_frames=2):
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor(shape_predictor_path)
        self.lStart, self.lEnd = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
        self.rStart, self.rEnd = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

        self.EYE_AR_THRESH = eye_ar_thresh
        self.EYE_AR_CONSEC_FRAMES = consec_frames

        self.COUNTER = 0
        self.TOTAL = 0
        self.BLINKED = False
        self.last_blink_time = 0
        self.ear_history = deque(maxlen=5)
        self.ear = 0  # Initialize the EAR value

    @staticmethod
    def eye_aspect_ratio(eye):
        A = np.linalg.norm(eye[1] - eye[5])
        B = np.linalg.norm(eye[2] - eye[4])
        C = np.linalg.norm(eye[0] - eye[3])
        return (A + B) / (2.0 * C)

    def process_frame(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        rects = self.detector(gray, 0)
        ear = None

        if rects:
            largest_rect = max(rects, key=lambda r: r.width() * r.height())
            shape = self.predictor(gray, largest_rect)
            shape = face_utils.shape_to_np(shape)
            leftEye = shape[self.lStart:self.lEnd]
            rightEye = shape[self.rStart:self.rEnd]
            ear = (self.eye_aspect_ratio(leftEye) + self.eye_aspect_ratio(rightEye)) / 2.0
            self.ear_history.append(round(ear, 3))

            current_time = time.time()
            if ear < self.EYE_AR_THRESH:
                self.COUNTER += 1
                if self.COUNTER >= self.EYE_AR_CONSEC_FRAMES and not self.BLINKED:
                    if current_time - self.last_blink_time >= 0.6:
                        self.TOTAL += 1
                        self.last_blink_time = current_time
                        self.BLINKED = True
            else:
                self.COUNTER = 0
                self.BLINKED = False

        self.ear = ear  # Set the ear value so it's available for GUI updates
        return ear, rects

    @staticmethod
    def find_available_camera(max_tested=5):
        for index in range(max_tested):
            cap = cv2.VideoCapture(index)
            if cap.isOpened():
                print(f"[INFO] Found available camera at index {index}")
                cap.release()
                return index
            cap.release()
        raise RuntimeError("No available camera found.")

class BlinkStatsLogger:
    def __init__(self, save_folder="saved_csv"):
        os.makedirs(save_folder, exist_ok=True)
        timestamp_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.csv_filename = os.path.join(save_folder, f"blinks_per_20s_{timestamp_str}.csv")

        with open(self.csv_filename, mode='w', newline='') as f:
            csv.writer(f).writerow(["Interval", "Blinks", "Rolling Average"])

        self.blinks_per_20s = []
        self.rolling_avgs = []
        self.start_time = time.time()
        self.interval_start = time.time()

    def log_interval(self, blinks_this_interval):
        interval_index = int((time.time() - self.start_time) // 20)
        self.blinks_per_20s.append(blinks_this_interval)
        if len(self.blinks_per_20s) > 10:
            self.blinks_per_20s.pop(0)

        avg = round(np.mean(self.blinks_per_20s), 2)
        self.rolling_avgs.insert(0, avg)
        if len(self.rolling_avgs) > 10:
            self.rolling_avgs.pop()

        with open(self.csv_filename, mode='a', newline='') as f:
            csv.writer(f).writerow([interval_index, blinks_this_interval, avg])

        return interval_index, avg

class VideoHandler:
    def __init__(self):
        self.vs = None
        self.fileStream = False

    def start(self, video_path="", camera_index=None):
        if video_path:
            print("[INFO] Starting video file stream...")
            self.vs = FileVideoStream(video_path).start()
            self.fileStream = True
        else:
            if camera_index is None:
                print("[INFO] Scanning for available camera...")
                camera_index = BlinkDetector.find_available_camera()
            print(f"[INFO] Starting video stream on camera {camera_index}...")
            self.vs = VideoStream(src=camera_index).start()
            self.fileStream = False
        time.sleep(1.0)

    def read(self):
        if self.fileStream and not self.vs.more():
            return None
        return self.vs.read()

    def stop(self):
        self.vs.stop()

    def is_file_stream(self):
        return self.fileStream
