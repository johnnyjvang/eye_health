import tkinter as tk
from tkinter import ttk
import threading
import time
import cv2
from imutils.video import VideoStream
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
from blink_utils import BlinkDetector, BlinkStatsLogger

# --- GUI ---
class App:
    def __init__(self, root):
        self.root = root
        self.root.title("Eye Blink Health Tracker")
        self.detector = BlinkDetector()

        # Camera selection
        self.available_cameras = self.detector.get_available_cameras()
        self.camera_var = tk.StringVar()

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
        text += f"\nBlinks in {self.detector.interval_sec}s: {blinks_this_interval}"

        rolling_avg = self.detector.rolling_avgs[0] if self.detector.rolling_avgs else 0
        text += f"\nRolling Average: {rolling_avg:.2f}"
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
