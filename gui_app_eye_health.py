import tkinter as tk
from tkinter import ttk
import threading
import time
import cv2
import dlib
import numpy as np
from imutils.video import VideoStream
from imutils import face_utils
from scipy.spatial import distance as dist
from collections import deque
from datetime import datetime
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# --- EAR Calculation ---
def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    return (A + B) / (2.0 * C)

# --- Detector Setup ---
predictor_path = "shape_predictor_68_face_landmarks.dat"  # Update path if needed
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

# --- Stats ---
EYE_AR_THRESH = 0.15
EYE_AR_CONSEC_FRAMES = 2

class BlinkDetector:
    def __init__(self):
        self.vs = None
        self.running = False
        self.COUNTER = 0
        self.TOTAL = 0
        self.BLINKED = False
        self.last_blink_time = 0
        self.blinks_per_interval = []
        self.rolling_avgs = []
        self.ear_history = deque(maxlen=5)
        self.interval_start = None
        self.interval_sec = 10  # Changed to 10s interval

        self.plot_x, self.plot_y = [], []

    def start(self, camera_index=0):
        self.vs = VideoStream(src=camera_index).start()
        time.sleep(1.0)
        self.running = True
        self.interval_start = time.time()

    def stop(self):
        self.running = False
        if self.vs:
            self.vs.stop()

    def process_frame(self):
        frame = self.vs.read()
        if frame is None:
            return None

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        rects = detector(gray, 0)

        ear = None
        if rects:
            largest_rect = max(rects, key=lambda r: r.width() * r.height())
            shape = predictor(gray, largest_rect)
            shape = face_utils.shape_to_np(shape)
            leftEye = shape[lStart:lEnd]
            rightEye = shape[rStart:rEnd]
            ear = (eye_aspect_ratio(leftEye) + eye_aspect_ratio(rightEye)) / 2.0
            self.ear_history.append(round(ear, 3))

            current_time = time.time()
            if ear < EYE_AR_THRESH:
                self.COUNTER += 1
                if self.COUNTER >= EYE_AR_CONSEC_FRAMES and not self.BLINKED:
                    if current_time - self.last_blink_time >= 1.0:
                        self.TOTAL += 1
                        self.last_blink_time = current_time
                        self.BLINKED = True
            else:
                self.COUNTER = 0
                self.BLINKED = False

        elapsed = time.time() - self.interval_start
        if elapsed >= self.interval_sec:
            blinks_this_interval = self.TOTAL
            self.TOTAL = 0
            self.interval_start = time.time()

            self.blinks_per_interval.append(blinks_this_interval)
            if len(self.blinks_per_interval) > 10:
                self.blinks_per_interval.pop(0)

            avg = round(np.mean(self.blinks_per_interval), 2)
            self.rolling_avgs.insert(0, avg)
            if len(self.rolling_avgs) > 10:
                self.rolling_avgs.pop()

            self.plot_x.append(len(self.plot_x))
            self.plot_y.append(blinks_this_interval)

        return ear

    def get_available_cameras(self):
        """Detect and return available camera indices."""
        available_cameras = []
        for index in range(5):  # Check the first 5 possible camera indices
            cap = cv2.VideoCapture(index)
            if cap.isOpened():
                available_cameras.append(index)
                cap.release()  # Release the camera after checking
        return available_cameras

# --- GUI ---
class App:
    def __init__(self, root):
        self.root = root
        self.root.title("Eye Blink Health Tracker")
        self.detector = BlinkDetector()

        # Get available cameras and populate ComboBox
        self.available_cameras = self.detector.get_available_cameras()
        self.camera_var = tk.StringVar()
        
        # Set the default camera to the first one in the list (if available)
        if self.available_cameras:
            self.camera_var.set(str(self.available_cameras[0]))
        else:
            self.camera_var.set("None")

        self.camera_label = tk.Label(root, text="Select Camera:")
        self.camera_label.pack(pady=5)

        self.camera_dropdown = ttk.Combobox(root, textvariable=self.camera_var, values=[str(i) for i in self.available_cameras])
        self.camera_dropdown.pack(pady=5)

        # Layout
        self.start_btn = ttk.Button(root, text="Start", command=self.start_detection)
        self.start_btn.pack(pady=5)
        self.stop_btn = ttk.Button(root, text="Stop", command=self.stop_detection)
        self.stop_btn.pack(pady=5)

        self.stats_label = tk.Label(root, text="Stats will appear here.", font=("Arial", 12))
        self.stats_label.pack(pady=10)

        # Plot
        self.fig, self.ax = plt.subplots(figsize=(5, 3))
        self.line, = self.ax.plot([], [], lw=2)
        self.ax.set_ylim(0, 40)
        self.ax.set_xlim(0, 1)
        self.ax.set_xlabel("Interval")
        self.ax.set_ylabel("Blinks")

        self.canvas = FigureCanvasTkAgg(self.fig, master=root)
        self.canvas.get_tk_widget().pack()

        self.updating = False

        # Handle window close event (safe shutdown)
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

    def start_detection(self):
        # Get selected camera index
        camera_index = int(self.camera_var.get())
        if not self.detector.running:
            self.detector.start(camera_index)
            self.updating = True
            self.update_gui()

    def stop_detection(self):
        self.detector.stop()
        self.updating = False

    def update_gui(self):
        if not self.detector.running:
            return

        ear = self.detector.process_frame()

        # Update stats label
        text = f"Running Total Blinks: {self.detector.TOTAL}"
        
        # Add debug print for blinks this interval
        blinks_this_interval = self.detector.blinks_per_interval[-1] if self.detector.blinks_per_interval else 0
        print(f"Blinks in 10s interval: {blinks_this_interval}")  # Debug print
        
        text += f"\nBlinks in {self.detector.interval_sec}s: {blinks_this_interval}"
        
        rolling_avg = self.detector.rolling_avgs[0] if self.detector.rolling_avgs else 0
        print(f"Rolling Average: {rolling_avg}")  # Debug print
        
        text += f"\nRolling Average: {rolling_avg}"
        text += f"\nCurrent EAR: {ear:.2f}" if ear else "Current EAR: --"

        self.stats_label.config(text=text)

        # Update plot only if new data is available
        if self.detector.plot_x and self.detector.plot_y:
            self.line.set_data(self.detector.plot_x, self.detector.plot_y)
            self.ax.set_xlim(0, max(self.detector.plot_x[-1], 1) if self.detector.plot_x else 1)
            self.canvas.draw()

        if self.updating:
            self.root.after(500, self.update_gui)  # update every 0.5 sec


    def on_close(self):
        """Handle safe shutdown of the camera when the window is closed"""
        self.detector.stop()
        self.root.quit()

# --- Run GUI ---
root = tk.Tk()
app = App(root)
root.mainloop()
