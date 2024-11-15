import tkinter as tk
from tkinter import filedialog
import cv2
from ultralytics import YOLO
import numpy as np
from PIL import Image, ImageTk

root = tk.Tk()
root.title("Détection et traitement avec YOLO")

# Initialiser YOLO
model = YOLO("yolov10n-face.pt")  # Chargez un modèle YOLO compatible

# Variables globales
video_capture = None
output_writer = None
scale_factor = 1.1
random_seed = 42
frame = None
processed_frame = None

# Charger une vidéo
def load_video():
    global video_capture
    filepath = filedialog.askopenfilename(filetypes=[("Video Files", "*.mp4;*.avi;*.mov")])
    if filepath:
        video_capture = cv2.VideoCapture(filepath)
        process_video()

# Traiter une vidéo
def process_video():
    global video_capture, processed_frame
    if not video_capture:
        return

    def update_frame():
        global frame, processed_frame
        ret, frame = video_capture.read()
        if not ret:
            video_capture.release()
            return
        
        processed_frame = frame.copy()
        results = model(frame)

        for box in results[0].boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            w, h = x2 - x1, y2 - y1
            x1 = max(0, int(x1 - (scale_factor - 1) * w / 2))
            y1 = max(0, int(y1 - (scale_factor - 1) * h / 2))
            x2 = min(frame.shape[1], int(x2 + (scale_factor - 1) * w / 2))
            y2 = min(frame.shape[0], int(y2 + (scale_factor - 1) * h / 2))
            
            roi = processed_frame[y1:y2, x1:x2]
            # Mélange des pixels de la boîte (par exemple)
            pixels = roi.reshape(-1, 3)
            np.random.seed(random_seed)
            np.random.shuffle(pixels)
            processed_frame[y1:y2, x1:x2] = pixels.reshape(roi.shape)

        show_frame(processed_frame)
        root.after(30, update_frame)

    update_frame()

# Sauvegarder une vidéo traitée
def save_video():
    global video_capture, output_writer, processed_frame
    if not video_capture:
        return

    output_filepath = filedialog.asksaveasfilename(defaultextension=".mp4",
                                                   filetypes=[("MP4 files", "*.mp4"),
                                                              ("AVI files", "*.avi")])
    if not output_filepath:
        return

    # Obtenir les propriétés de la vidéo
    frame_width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(video_capture.get(cv2.CAP_PROP_FPS))

    # Initialiser VideoWriter
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    output_writer = cv2.VideoWriter(output_filepath, fourcc, fps, (frame_width, frame_height))

    video_capture.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Revenir au début de la vidéo

    while True:
        ret, frame = video_capture.read()
        if not ret:
            break

        processed_frame = frame.copy()
        results = model(frame)
        for box in results[0].boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            w, h = x2 - x1, y2 - y1
            x1 = max(0, int(x1 - (scale_factor - 1) * w / 2))
            y1 = max(0, int(y1 - (scale_factor - 1) * h / 2))
            x2 = min(frame.shape[1], int(x2 + (scale_factor - 1) * w / 2))
            y2 = min(frame.shape[0], int(y2 + (scale_factor - 1) * h / 2))
            
            roi = processed_frame[y1:y2, x1:x2]
            pixels = roi.reshape(-1, 3)
            np.random.seed(random_seed)
            np.random.shuffle(pixels)
            processed_frame[y1:y2, x1:x2] = pixels.reshape(roi.shape)

        output_writer.write(processed_frame)

    output_writer.release()
    video_capture.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Revenir au début

# Afficher une image dans Tkinter
def show_frame(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(image)
    img_tk = ImageTk.PhotoImage(img_pil)
    label_video.config(image=img_tk)
    label_video.image = img_tk

# Widgets de l'interface
btn_load_video = tk.Button(root, text="Charger une vidéo", command=load_video)
btn_load_video.pack()

btn_save_video = tk.Button(root, text="Sauvegarder la vidéo", command=save_video)
btn_save_video.pack()

label_video = tk.Label(root)
label_video.pack()

# Lancer l'application

root.mainloop()
