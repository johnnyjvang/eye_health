import argparse
import time
import imutils
from imutils.video import VideoStream, FileVideoStream
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk
from tkinter import Label
from blink_utils import BlinkDetector, BlinkStatsLogger

# --- Argparser ---
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--shape-predictor", required=True, help="path to facial landmark predictor")
ap.add_argument("-v", "--video", type=str, default="", help="path to input video file")
ap.add_argument("-c", "--camera", type=int, default=0, help="camera index")
args = vars(ap.parse_args())

# --- Init classes ---
detector = BlinkDetector(args["shape_predictor"])
stats_logger = BlinkStatsLogger()

# --- Video setup ---
print("[INFO] starting video stream...")
vs = VideoStream(src=args["camera"]).start() if not args["video"] else FileVideoStream(args["video"]).start()
fileStream = bool(args["video"])
time.sleep(1.0)

# --- Matplotlib Plot ---
plt.style.use("seaborn-v0_8")
fig, ax = plt.subplots()
x_vals, y_vals = [], []
line, = ax.plot([], [], lw=2)
ax.set_ylim(0, 40)  # Set a reasonable upper limit for blinks
ax.set_xlim(0, 1)
ax.set_xlabel("10-sec Interval")
ax.set_ylabel("Blinks")
ax.set_title("Real-Time Blink Count (10s intervals)")

def update_plot(i):
    line.set_data(x_vals, y_vals)
    ax.set_xlim(0, max(x_vals[-1], 1) if x_vals else 1)  # Adjust x-axis dynamically
    return [line]

# Create Tkinter Window
root = tk.Tk()
root.title("Real-Time Blink Detection")
root.geometry("800x600")

# Labels for displaying blink data
blink_label = Label(root, text="Blinks: 0", font=("Helvetica", 16))
blink_label.pack()

ear_label = Label(root, text="EAR: --", font=("Helvetica", 16))
ear_label.pack()

avg_blink_label = Label(root, text="Avg Blinks/10s: --", font=("Helvetica", 16))
avg_blink_label.pack()

# --- Matplotlib Canvas for Tkinter
canvas = FigureCanvasTkAgg(fig, master=root)  # A tk.DrawingArea
canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
canvas.draw()

def update_gui():
    global blink_label, ear_label, avg_blink_label
    
    # Update the GUI labels
    ear_label.config(text=f"EAR: {detector.ear:.2f}" if detector.ear else "EAR: --")
    blink_label.config(text=f"Blinks: {detector.TOTAL}")
    avg_blinks = stats_logger.rolling_avgs[0] if stats_logger.rolling_avgs else 0
    avg_blink_label.config(text=f"Avg Blinks/10s: {avg_blinks:.2f}")
    
    # Update the plot
    update_plot(None)
    canvas.draw()

# --- Main loop ---
interval_blinks = 0
interval_start = time.time()

def process_frame():
    global interval_blinks, interval_start

    try:
        if fileStream and not vs.more():
            root.quit()

        frame = vs.read()
        if frame is None:
            return

        frame = imutils.resize(frame, width=450)

        # Process the frame
        ear, rects = detector.process_frame(frame)

        if rects:
            interval_blinks = detector.TOTAL

        elapsed = int(time.time() - interval_start)
        
        # Change the interval to 10 seconds for logging
        if elapsed >= 10:
            interval_index, avg = stats_logger.log_interval(interval_blinks)
            x_vals.append(interval_index)
            y_vals.append(interval_blinks)

            print(f"[INFO] Interval {interval_index}: {interval_blinks} blinks | Avg: {avg}")

            interval_blinks = 0
            detector.TOTAL = 0
            interval_start = time.time()

        # Call update_gui to refresh the labels and plot
        update_gui()

        # Refresh the GUI every 100 ms (for smoother experience)
        root.after(100, process_frame)

    except KeyboardInterrupt:
        print("[INFO] Interrupted by user.")
        root.quit()

# Start the frame processing loop
process_frame()

# --- Start Tkinter mainloop ---
root.mainloop()

# Cleanup
vs.stop()
plt.close(fig)
