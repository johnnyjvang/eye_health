# pip install dlib opencv-python numpy imutils scipy matplotlib seaborn
# Reference 1: https://github.com/rmenoli/Eye-blinking-SVM
# Reference 2: https://github.com/Practical-CV/EYE-BLINK-DETECTION-WITH-OPENCV-AND-DLIB
# Blog: https://pyimagesearch.com/2017/04/24/eye-blink-detection-opencv-python-dlib/
# python3 eye_health_tracker.py --shape-predictor shape_predictor_68_face_landmarks.dat --camera 0

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
import os
from datetime import datetime
from collections import deque  # Added for EAR history

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
EYE_AR_THRESH = 0.18
EYE_AR_CONSEC_FRAMES = 2
COUNTER = 0
TOTAL = 0
BLINKED = False
last_blink_time = 0
blinks_per_20s = []
rolling_avgs = []
ear_history = deque(maxlen=5)  # Keep last 5 EAR values

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

# --- Create save folder ---
save_folder = "saved_csv"
os.makedirs(save_folder, exist_ok=True)

# --- Timing ---
start_time = time.time()
interval_start = time.time()
second_timer = 0

# --- Create filename with timestamp ---
timestamp_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
csv_filename = os.path.join(save_folder, f"blinks_per_20s_{timestamp_str}.csv")

# --- CSV Init ---
with open(csv_filename, mode='w', newline='') as f:
    csv.writer(f).writerow(["Interval", "Blinks", "Rolling Average"])

# --- Matplotlib Plot ---
plt.style.use("seaborn-v0_8")  # Ubuntu 24.04 style fix
fig, ax = plt.subplots()
x_vals, y_vals = [], []
line, = ax.plot([], [], lw=2)
ax.set_ylim(0, 40)
ax.set_xlim(0, 1)
ax.set_xlabel("20-sec Interval")
ax.set_ylabel("Blinks")
ax.set_title("Real-Time Blink Count (20s intervals)")

def update_plot(i):
    line.set_data(x_vals, y_vals)
    ax.set_xlim(0, max(x_vals[-1], 1) if x_vals else 1)
    return [line]

ani = FuncAnimation(fig, update_plot, interval=1000, blit=True)

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

        if rects:
            largest_rect = max(rects, key=lambda r: r.width() * r.height())
            shape = predictor(gray, largest_rect)
            shape = face_utils.shape_to_np(shape)
            leftEye = shape[lStart:lEnd]
            rightEye = shape[rStart:rEnd]
            ear = (eye_aspect_ratio(leftEye) + eye_aspect_ratio(rightEye)) / 2.0

            ear_history.append(round(ear, 3))  # Update EAR history

            # Draw contours
            cv2.drawContours(frame, [cv2.convexHull(leftEye)], -1, (0, 255, 0), 1)
            cv2.drawContours(frame, [cv2.convexHull(rightEye)], -1, (0, 255, 0), 1)

            # Updated Blink detection with time threshold
            current_time = time.time()

            if ear < EYE_AR_THRESH:
                COUNTER += 1
                if COUNTER >= EYE_AR_CONSEC_FRAMES and not BLINKED:
                    if current_time - last_blink_time >= 0.6:
                        TOTAL += 1
                        last_blink_time = current_time
                        BLINKED = True
                        print(f"[INFO] Blink detected at {datetime.now().strftime('%H:%M:%S')}")
            else:
                COUNTER = 0
                BLINKED = False

        # --- Timer and Stats ---
        elapsed = int(time.time() - interval_start)
        if elapsed >= 20:
            interval_index = int((time.time() - start_time) // 20)
            blinks_this_interval = TOTAL
            TOTAL = 0
            interval_start = time.time()
            second_timer = 0

            blinks_per_20s.append(blinks_this_interval)
            if len(blinks_per_20s) > 10:
                blinks_per_20s.pop(0)

            avg = round(np.mean(blinks_per_20s), 2)
            rolling_avgs.insert(0, avg)
            if len(rolling_avgs) > 10:
                rolling_avgs.pop()

            x_vals.append(interval_index)
            y_vals.append(blinks_this_interval)

            with open(csv_filename, mode='a', newline='') as f:
                csv.writer(f).writerow([interval_index, blinks_this_interval, avg])

            print(f"[INFO] Interval {interval_index}: {blinks_this_interval} blinks | Avg: {avg}")

        else:
            second_timer = elapsed

        # --- Overlay Stats ---
        cv2.putText(frame, f"EAR: {ear:.2f}" if rects else "EAR: --", (300, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        cv2.putText(frame, f"Blinks: {TOTAL}", (10, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.putText(frame, f"Timer: {second_timer}s", (10, 45),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 0), 2)

        # Display all rolling averages side by side
        for i, val in enumerate(rolling_avgs):
            cv2.putText(frame, f"{val}", (10 + i * 45, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 0, 255), 2)

        # Display last 5 EAR values
        cv2.putText(frame, "Last 5 EAR:", (300, 45),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 100), 1)
        for i, val in enumerate(reversed(ear_history)):
            cv2.putText(frame, f"{val}", (300, 65 + i * 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 100, 0), 1)

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
