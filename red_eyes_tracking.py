import cv2
import numpy as np
import argparse
import time
from collections import deque

# CLI input for camera index
parser = argparse.ArgumentParser()
parser.add_argument('--camera', type=int, default=0, help='Camera index')
args = parser.parse_args()

# Load Haar cascade
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
cap = cv2.VideoCapture(args.camera)

# Queues for smoothing + 20-second logs (L and R)
left_history = deque(maxlen=10)
right_history = deque(maxlen=10)
left_log = []
right_log = []

def classify_redness(ratio):
    if ratio < 0.05:
        return "Normal"
    elif ratio < 0.10:
        return "Tired"
    else:
        return "Likely Irritated"

def compute_redness(hsv_eye, w, h):
    lower_red1 = np.array([0, 70, 50])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 70, 50])
    upper_red2 = np.array([180, 255, 255])

    mask1 = cv2.inRange(hsv_eye, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv_eye, lower_red2, upper_red2)
    mask = mask1 + mask2

    red_area = np.sum(mask > 0)
    eye_area = w * h
    return red_area / eye_area if eye_area > 0 else 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    eyes = eye_cascade.detectMultiScale(gray, 1.3, 5)

    eyes = sorted(eyes, key=lambda e: e[0])  # Sort left to right

    if len(eyes) >= 2:
        (lx, ly, lw, lh), (rx, ry, rw, rh) = eyes[:2]

        # Process left eye
        left_eye = frame[ly:ly+lh, lx:lx+lw]
        left_hsv = cv2.cvtColor(left_eye, cv2.COLOR_BGR2HSV)
        l_ratio = compute_redness(left_hsv, lw, lh)
        left_history.append(l_ratio)
        l_smoothed = np.mean(left_history)
        left_log.append((time.time(), l_smoothed))
        left_log = [(t, v) for t, v in left_log if time.time() - t <= 20]
        l_avg = np.mean([v for _, v in left_log])
        l_status = classify_redness(l_avg)

        # Process right eye
        right_eye = frame[ry:ry+rh, rx:rx+rw]
        right_hsv = cv2.cvtColor(right_eye, cv2.COLOR_BGR2HSV)
        r_ratio = compute_redness(right_hsv, rw, rh)
        right_history.append(r_ratio)
        r_smoothed = np.mean(right_history)
        right_log.append((time.time(), r_smoothed))
        right_log = [(t, v) for t, v in right_log if time.time() - t <= 20]
        r_avg = np.mean([v for _, v in right_log])
        r_status = classify_redness(r_avg)

        # Draw rectangles
        cv2.rectangle(frame, (lx, ly), (lx+lw, ly+lh), (0, 255, 0), 2)
        cv2.rectangle(frame, (rx, ry), (rx+rw, ry+rh), (0, 255, 0), 2)

        # Bottom overlay display
        height, width = frame.shape[:2]
        cv2.putText(frame, f"Left Avg: {l_avg:.2%} | {l_status}", (10, height - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 100, 100), 2)
        cv2.putText(frame, f"Right Avg: {r_avg:.2%} | {r_status}", (10, height - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 100, 255), 2)

    cv2.imshow("Eye Redness Monitor", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
