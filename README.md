Eye Blink Detection with Dlib, OpenCV, and Python

This project detects and counts eye blinks in real-time using computer vision techniques. It leverages facial landmark detection to calculate the Eye Aspect Ratio (EAR), which is used to infer blink events. The program also tracks blink frequency over 20-second intervals and saves the data as a CSV file for further analysis, along with real-time visualization using Matplotlib.

Installation
- pip install dlib opencv-python numpy imutils scipy matplotlib seaborn

Script: 
- python3 blink_detection_final.py --shape-predictor shape_predictor_68_face_landmarks.dat --camera 0
- python3 blink_detection_final.py --shape-predictor shape_predictor_68_face_landmarks.dat --video path_to_video.mp4

Output
- A live window showing the blink detection with EAR stats and rolling averages.
- A CSV file with blink counts per 20-second interval saved in /saved_csv.

Features
- Real-time eye blink detection via webcam or video file.
- Eye Aspect Ratio (EAR) calculation based on dlib's 68-point facial landmark detector.
- Logs blink counts every 20 seconds and computes rolling averages.
- Saves blink data to timestamped CSV files.
- Live plot of blink frequency using Matplotlib.

Technologies Used
- Python 3
- dlib
- OpenCV
- NumPy
- SciPy
- Imutils
- Matplotlib
- Seaborn

References
- Eye-blinking-SVM: https://github.com/rmenoli/Eye-blinking-SVM
- Eye Blink Detection with OpenCV, Python, and dlib (PyImageSearch): https://pyimagesearch.com/2017/04/24/eye-blink-detection-opencv-python-dlib/
- Practical-CV Eye Blink Detection: https://github.com/Practical-CV/EYE-BLINK-DETECTION-WITH-OPENCV-AND-DLIB
