import cv2
import dlib
import numpy as np
from imutils import face_utils
from scipy.spatial import distance as dist
import threading
import tkinter as tk
from tkinter import messagebox

# EAR function
def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    return (A + B) / (2.0 * C)

# Calculate distance (proxy using face height)
def calculate_distance(face_height):
    KNOWN_FACE_WIDTH = 14.0  # cm
    FOCAL_LENGTH = 500
    if face_height == 0:
        return 0
    distance = (KNOWN_FACE_WIDTH * FOCAL_LENGTH) / face_height
    return distance

class EARApp:
    def __init__(self, root):
        self.root = root
        self.root.title("EAR & Distance Measurement")
        self.running = False

        self.start_button = tk.Button(root, text="Start", command=self.start)
        self.start_button.pack(pady=5)

        self.stop_button = tk.Button(root, text="Stop", command=self.stop, state=tk.DISABLED)
        self.stop_button.pack(pady=5)

        self.exit_button = tk.Button(root, text="Exit", command=self.on_exit)
        self.exit_button.pack(pady=5)

        self.info_text = tk.Text(root, height=15, width=50)
        self.info_text.pack(pady=10)

        self.thread = None

    def start(self):
        if not self.running:
            self.running = True
            self.start_button.config(state=tk.DISABLED)
            self.stop_button.config(state=tk.NORMAL)
            self.info_text.insert(tk.END, "[INFO] Started monitoring...\n")
            self.thread = threading.Thread(target=self.process_loop)
            self.thread.start()

    def stop(self):
        if self.running:
            self.running = False
            self.start_button.config(state=tk.NORMAL)
            self.stop_button.config(state=tk.DISABLED)
            self.info_text.insert(tk.END, "[INFO] Stopped monitoring.\n")

    def process_loop(self):
        detector = dlib.get_frontal_face_detector()
        predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

        (lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
        (rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

        cap = cv2.VideoCapture(0)

        while self.running:
            ret, frame = cap.read()
            if not ret:
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            rects = detector(gray, 0)

            for rect in rects:
                shape = predictor(gray, rect)
                shape = face_utils.shape_to_np(shape)

                leftEye = shape[lStart:lEnd]
                rightEye = shape[rStart:rEnd]
                ear = (eye_aspect_ratio(leftEye) + eye_aspect_ratio(rightEye)) / 2.0

                face_height = rect.bottom() - rect.top()
                distance = calculate_distance(face_height)

                info = f"EAR: {ear:.2f} | Face Height: {face_height}px | Distance: {distance:.2f}cm\n"
                self.info_text.insert(tk.END, info)
                self.info_text.see(tk.END)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

    def on_exit(self):
        if self.running:
            self.running = False
        self.root.quit()
        self.root.destroy()

# Run GUI
root = tk.Tk()
app = EARApp(root)
root.mainloop()
