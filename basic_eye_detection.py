#python3 basic_eye_detection.py --shape-predictor shape_predictor_68_face_landmarks.dat --camera 2

import argparse
import time
import dlib
import cv2
import numpy as np
import csv
from imutils.video import FileVideoStream, VideoStream
from imutils import face_utils
from scipy.spatial import distance as dist
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import imutils

# --- EAR Calculation ---
def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    return (A + B) / (2.0 * C)

# --- Argparser ---
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--shape-predictor", required=True, help="path to facial landmark predictor")
ap.add_argument("-v", "--video", type=str, default="", help="path to input video file")
ap.add_argument("-c", "--camera", type=int, default=0, help="camera index")
args = vars(ap.parse_args())

# --- Constants ---
EYE_AR_THRESH = 0.25
EYE_AR_CONSEC_FRAMES = 2
COUNTER = 0
TOTAL = 0
BLINKED = False
blinks_per_minute = []
timestamps = []

# --- Setup ---
print("[INFO] loading predictor...")
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(args["shape_predictor"])
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

print("[INFO] starting video stream...")
vs = FileVideoStream(args["video"]).start() if args["video"] else VideoStream(src=args["camera"]).start()
fileStream = bool(args["video"])
time.sleep(1.0)

# --- Timing ---
start_time = time.time()
minute_start = time.time()
second_timer = 0
csv_filename = "blinks_per_minute.csv"

# --- CSV Init ---
with open(csv_filename, mode='w', newline='') as f:
    csv.writer(f).writerow(["Minute", "Blinks"])

# --- Matplotlib Plot ---
plt.style.use("seaborn")
fig, ax = plt.subplots()
x_vals, y_vals = [], []
line, = ax.plot([], [], lw=2)
ax.set_ylim(0, 40)
ax.set_xlim(0, 1)
ax.set_xlabel("Minute")
ax.set_ylabel("Blinks")
ax.set_title("Real-Time Blink Count")
ani = FuncAnimation(fig, lambda i: line.set_data(x_vals, y_vals), interval=1000, blit=True)

# --- Main Loop ---
try:
    while True:
        if fileStream and not vs.more():
            break

        frame = vs.read()
        if frame is None:
            continue

        frame = imutils.resize(frame, width=450)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        rects = detector(gray, 0)

        for rect in rects:
            shape = predictor(gray, rect)
            shape = face_utils.shape_to_np(shape)
            leftEye, rightEye = shape[lStart:lEnd], shape[rStart:rEnd]
            ear = (eye_aspect_ratio(leftEye) + eye_aspect_ratio(rightEye)) / 2.0

            # Draw eye contours
            cv2.drawContours(frame, [cv2.convexHull(leftEye)], -1, (0, 255, 0), 1)
            cv2.drawContours(frame, [cv2.convexHull(rightEye)], -1, (0, 255, 0), 1)

            # Blink Detection Logic
            if ear < EYE_AR_THRESH:
                COUNTER += 1
            else:
                if COUNTER >= EYE_AR_CONSEC_FRAMES and not BLINKED:
                    TOTAL += 1
                    BLINKED = True
                COUNTER = 0

            if ear >= EYE_AR_THRESH:
                BLINKED = False

        # --- Timer and Stats ---
        elapsed = int(time.time() - minute_start)
        if elapsed >= 60:
            minute_index = int((time.time() - start_time) // 60)
            blinks_this_minute = TOTAL
            TOTAL = 0
            minute_start = time.time()
            second_timer = 0

            blinks_per_minute.append(blinks_this_minute)
            timestamps.append(minute_index)
            x_vals.append(minute_index)
            y_vals.append(blinks_this_minute)

            with open(csv_filename, mode='a', newline='') as f:
                csv.writer(f).writerow([minute_index, blinks_this_minute])

            print(f"[INFO] Minute {minute_index}: {blinks_this_minute} blinks")

        else:
            second_timer = elapsed

        avg_bpm = blinks_per_minute[-1] if blinks_per_minute else 0

        # --- Overlay Stats ---
        cv2.putText(frame, f"EAR: {ear:.2f}", (300, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        cv2.putText(frame, f"Blinks: {TOTAL}", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.putText(frame, f"Timer: {second_timer}s", (10, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 0), 2)
        cv2.putText(frame, f"Avg BPM: {avg_bpm}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 0, 255), 2)

        # --- Display Frame ---
        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

except KeyboardInterrupt:
    print("[INFO] Interrupted by user.")

# --- Cleanup ---
cv2.destroyAllWindows()
vs.stop()
plt.close(fig)
