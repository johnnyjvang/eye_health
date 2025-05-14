# pip install opencv-python dlib numpy imutils matplotlib
# python measure_ear.py --shape-predictor shape_predictor_68_face_landmarks.dat

'''
The shape_predictor_68_face_landmarks.dat file is a pre-trained model provided by dlib for detecting facial landmarks in an image or video stream.

Purpose:
This model allows you to detect specific facial features, such as the eyes, nose, mouth, and jawline, by predicting 68 key points (landmarks) on the face. These points are used for tasks like:

    - Facial landmark detection: Identifying the locations of eyes, eyebrows, nose, and mouth.
    - Eye aspect ratio (EAR) calculation: Measuring the eye shape to determine blinks or gaze direction.
    - Face alignment: Adjusting the face image to a standard orientation.

How it works:
    - Input: You feed the model an image (typically grayscale).
    - Output: The model returns the coordinates of 68 landmarks, which are specific points on the face.

For example:

    - Left eye landmarks: Points 36–41
    - Right eye landmarks: Points 42–47
    - Nose landmarks: Points 27–35
    - Mouth landmarks: Points 48–67

'''

import cv2
import dlib
import numpy as np
from imutils import face_utils
from scipy.spatial import distance as dist
import matplotlib.pyplot as plt

# EAR function
def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    return (A + B) / (2.0 * C)

# Initialize face detector and predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

# Data holders
ear_list = []
distance_list = []
face_height_list = []

# Video stream
cap = cv2.VideoCapture(0)

print("[INFO] Press 's' to save EAR, face height, and face distance, 'q' to quit.")
print("[INFO] Vary your distance from the camera for each sample.")

# Calibration parameters (example)
KNOWN_FACE_WIDTH = 14.0  # Known face width in cm (real-world average width of a human face)
FOCAL_LENGTH = 500  # Focal length (can be calibrated based on camera settings)

def calculate_distance(face_height):
    """
    Calculate the distance to the face using the face bounding box height (proxy for size).
    """
    # Assuming the face height is inversely proportional to the distance
    if face_height == 0:
        return 0
    # Use a basic linear relationship for the example
    distance = (KNOWN_FACE_WIDTH * FOCAL_LENGTH) / face_height
    return distance

while True:
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

        # Calculate face bounding box height (proxy for distance)
        face_height = rect.bottom() - rect.top()

        # Calculate the distance using the bounding box height
        distance = calculate_distance(face_height)

        # Draw eye landmarks and face box
        cv2.drawContours(frame, [cv2.convexHull(leftEye)], -1, (0, 255, 0), 1)
        cv2.drawContours(frame, [cv2.convexHull(rightEye)], -1, (0, 255, 0), 1)
        cv2.rectangle(frame, (rect.left(), rect.top()), (rect.right(), rect.bottom()), (255, 0, 0), 2)

        # Display EAR, face height, and distance
        cv2.putText(frame, f"EAR: {ear:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(frame, f"Face Height: {face_height} px", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(frame, f"Distance: {distance:.2f} cm", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    cv2.imshow("EAR vs Distance", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('s') and rects:
        # Save the current EAR, face height, and distance
        ear_list.append(ear)
        distance_list.append(distance)
        face_height_list.append(face_height)
        print(f"[INFO] Saved EAR: {ear:.2f}, Face Height: {face_height}, Distance: {distance:.2f} cm")
    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# --- Plotting the results ---
if ear_list:
    plt.figure(figsize=(8, 6))

    # Plot EAR vs Distance
    plt.subplot(1, 2, 1)
    plt.scatter(distance_list, ear_list, c='blue', label='Data points')
    # Fit a linear regression line
    coeffs = np.polyfit(distance_list, ear_list, 1)
    poly1d_fn = np.poly1d(coeffs)
    plt.plot(distance_list, poly1d_fn(distance_list), '--r', label='Linear Fit')
    plt.title("EAR vs Face Distance")
    plt.xlabel("Distance to Face (cm)")
    plt.ylabel("EAR")
    plt.legend()
    plt.grid(True)

    # Plot EAR vs Face Height
    plt.subplot(1, 2, 2)
    plt.scatter(face_height_list, ear_list, c='green', label='Data points')
    # Fit a linear regression line
    coeffs2 = np.polyfit(face_height_list, ear_list, 1)
    poly1d_fn2 = np.poly1d(coeffs2)
    plt.plot(face_height_list, poly1d_fn2(face_height_list), '--r', label='Linear Fit')
    plt.title("EAR vs Face Height (pixels)")
    plt.xlabel("Face Height (pixels)")
    plt.ylabel("EAR")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()

    print(f"[INFO] Fitted model for EAR vs Distance: EAR = {coeffs[0]:.4f} * Distance + {coeffs[1]:.4f}")
    print(f"[INFO] Fitted model for EAR vs Face Height: EAR = {coeffs2[0]:.4f} * Face Height + {coeffs2[1]:.4f}")
else:
    print("[INFO] No data collected.")
