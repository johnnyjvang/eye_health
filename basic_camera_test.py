import cv2
import sys

def window_exists(window_name):
    return cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) >= 1

if len(sys.argv) != 2:
    print("Usage: python3 camera_viewer.py <camera_index>")
    sys.exit(1)

try:
    cam_index = int(sys.argv[1])
except ValueError:
    print("Camera index must be an integer.")
    sys.exit(1)

cap = cv2.VideoCapture(cam_index)

if not cap.isOpened():
    print(f"Cannot open camera at index {cam_index}")
    sys.exit(1)

window_name = 'Camera Feed'
print(f"Camera {cam_index} is now active. Press 'q' or close the window to quit.")

cv2.namedWindow(window_name)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Can't receive frame. Exiting...")
        break

    cv2.imshow(window_name, frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    if not window_exists(window_name):
        break

cap.release()
cv2.destroyAllWindows()
