import tkinter as tk
from tkinter import *
from tkinter import filedialog
from tkinter import ttk
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
random_seed = 4
frame = None
processed_frame = None
track = tk.IntVar()
trackers = []

# Charger une vidéo
def load_video():
    global video_capture, frame, processed_frame
    filepath = filedialog.askopenfilename(defaultextension=".mp4",
                                          filetypes=[("MP4 files", "*.mp4"),
                                                     ("AVI files", "*.avi")])
    if filepath:
        video_capture = cv2.VideoCapture(filepath)
        ret, frame = video_capture.read()
        processed_frame = frame.copy()
        results = model(frame)
        i = 0

        for box in results[0].boxes:
            i = i+1
            x1, y1, x2, y2 = map(int, box.xyxy[0])  # Obtenir les coordonnées de la boîte
            #boxes.append([x1, y1, x2, y2])

            # Créer un tracker pour chaque visage détecté
            tracker = cv2.TrackerCSRT_create()

            trackers.append(tracker)
            tracker.init(frame, (x1, y1, x2 - x1, y2 - y1)) 

            # Dessiner la boîte sur l'image
            cv2.rectangle(processed_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Dessiner en vert
            cv2.putText(processed_frame, f"{i}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        show_frame(processed_frame)

# Traiter une vidéo
def shuffle_video():
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

    video_capture.set(cv2.CAP_PROP_POS_FRAMES, 0)

    def update_frameS():
        global frame, processed_frame
        ret, frame = video_capture.read()
        
        if not ret:
            video_capture.release()
            output_writer.release()
            print("Processing complete.")

            return

        processed_frame = frame.copy()

        if track.get() == 1 :        
            if int(user_Entry.get()) > 0:
                success, bbox = trackers[int(user_Entry.get())-1].update(processed_frame)
                if success:
                    x1, y1, w, h = map(int, bbox)
                    x2, y2 = x1 + w, y1 + h

                    # Extraire la région d'intérêt
                    roi = processed_frame[y1:y2, x1:x2]

                    # Mélanger les pixels dans la boîte détectée
                    pixels = roi.reshape(-1, 3)
                    np.random.shuffle(pixels)
                    roi_mixed = pixels.reshape(roi.shape)

                    # Remettre la ROI mélangée dans l'image originale
                    processed_frame[y1:y2, x1:x2] = roi_mixed        
            else:
                for i, tracker in enumerate(trackers):
                    success, bbox = tracker.update(processed_frame)
                    if success:
                        x1, y1, w, h = map(int, bbox)
                        x2, y2 = x1 + w, y1 + h

                        # Extraire la région d'intérêt
                        roi = processed_frame[y1:y2, x1:x2]

                        # Mélanger les pixels dans la boîte détectée
                        pixels = roi.reshape(-1, 3)
                        np.random.shuffle(pixels)
                        roi_mixed = pixels.reshape(roi.shape)

                        # Remettre la ROI mélangée dans l'image originale
                        processed_frame[y1:y2, x1:x2] = roi_mixed
        else :      
            results = model(frame)  
            if int(user_Entry.get()) > 0:
                boxes = results[0].boxes
                box = boxes[int(user_Entry.get())-1]
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
            else :
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
        root.after(30, update_frameS)
        output_writer.write(processed_frame)

    update_frameS()
    

def blur_video():
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

    video_capture.set(cv2.CAP_PROP_POS_FRAMES, 0)

    def update_frameB():
        global frame, processed_frame
        ret, frame = video_capture.read()
        
        if not ret:
            video_capture.release()
            output_writer.release()
            print("Processing complete.")

            return

        processed_frame = frame.copy()

        if track.get() == 1 :        
            if int(user_Entry.get()) > 0:
                success, bbox = trackers[int(user_Entry.get())-1].update(processed_frame)
                if success:
                    x1, y1, w, h = map(int, bbox)
                    x2, y2 = x1 + w, y1 + h

                    roi = processed_frame[y1:y2, x1:x2]
                    # Mélange des pixels de la boîte (par exemple)
                    blurred_roi = cv2.GaussianBlur(roi, (51, 51), 0)  
            
                    processed_frame[(y1):(y2), (x1):(x2)] = blurred_roi       
            else:
                for i, tracker in enumerate(trackers):
                    success, bbox = tracker.update(processed_frame)
                    if success:
                        x1, y1, w, h = map(int, bbox)
                        x2, y2 = x1 + w, y1 + h

                        # Extraire la région d'intérêt
                        roi = processed_frame[y1:y2, x1:x2]
                        # Mélange des pixels de la boîte (par exemple)
                        blurred_roi = cv2.GaussianBlur(roi, (51, 51), 0)  
                
                        processed_frame[(y1):(y2), (x1):(x2)] = blurred_roi
        else :      
            results = model(frame)  
            if int(user_Entry.get()) > 0:
                boxes = results[0].boxes
                box = boxes[int(user_Entry.get())-1]
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                w, h = x2 - x1, y2 - y1
                x1 = max(0, int(x1 - (scale_factor - 1) * w / 2))
                y1 = max(0, int(y1 - (scale_factor - 1) * h / 2))
                x2 = min(frame.shape[1], int(x2 + (scale_factor - 1) * w / 2))
                y2 = min(frame.shape[0], int(y2 + (scale_factor - 1) * h / 2))
                
                roi = processed_frame[y1:y2, x1:x2]
                # Mélange des pixels de la boîte (par exemple)
                blurred_roi = cv2.GaussianBlur(roi, (51, 51), 0)  
        
                processed_frame[(y1):(y2), (x1):(x2)] = blurred_roi
            else :
                for box in results[0].boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    w, h = x2 - x1, y2 - y1
                    x1 = max(0, int(x1 - (scale_factor - 1) * w / 2))
                    y1 = max(0, int(y1 - (scale_factor - 1) * h / 2))
                    x2 = min(frame.shape[1], int(x2 + (scale_factor - 1) * w / 2))
                    y2 = min(frame.shape[0], int(y2 + (scale_factor - 1) * h / 2))
                    
                    roi = processed_frame[y1:y2, x1:x2]
                    # Mélange des pixels de la boîte (par exemple)
                    blurred_roi = cv2.GaussianBlur(roi, (51, 51), 0)  
            
                    processed_frame[(y1):(y2), (x1):(x2)] = blurred_roi

        show_frame(processed_frame)
        root.after(30, update_frameB)
        output_writer.write(processed_frame)

    update_frameB()
    

def pixel_video():
    
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

    video_capture.set(cv2.CAP_PROP_POS_FRAMES, 0)

    def update_frameP():
        global frame, processed_frame
        ret, frame = video_capture.read()
        if not ret:
            video_capture.release()
            output_writer.release()
            return
        
        processed_frame = frame.copy()
        if track.get() == 1 :        
            if int(user_Entry.get()) > 0:
                success, bbox = trackers[int(user_Entry.get())-1].update(processed_frame)
                if success:
                    x1, y1, w, h = map(int, bbox)
                    x2, y2 = x1 + w, y1 + h

                    roi = processed_frame[y1:y2, x1:x2]
                    # Mélange des pixels de la boîte (par exemple)
                    temp = cv2.resize(roi, (4, 4), interpolation=cv2.INTER_LINEAR)
                    pixel_roi = cv2.resize(temp, (x2 - x1, y2 - y1), interpolation=cv2.INTER_NEAREST)   
                    processed_frame[(y1):(y2), (x1):(x2)] = pixel_roi  
            else:
                for i, tracker in enumerate(trackers):
                    success, bbox = tracker.update(processed_frame)
                    if success:
                        x1, y1, w, h = map(int, bbox)
                        x2, y2 = x1 + w, y1 + h

                        roi = processed_frame[y1:y2, x1:x2]
                        # Mélange des pixels de la boîte (par exemple)
                        temp = cv2.resize(roi, (4, 4), interpolation=cv2.INTER_LINEAR)
                        pixel_roi = cv2.resize(temp, (x2 - x1, y2 - y1), interpolation=cv2.INTER_NEAREST) 
                        processed_frame[(y1):(y2), (x1):(x2)] = pixel_roi
        else :      
            results = model(frame)  
            if int(user_Entry.get()) > 0:
                boxes = results[0].boxes
                box = boxes[int(user_Entry.get())-1]
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                w, h = x2 - x1, y2 - y1
                x1 = max(0, int(x1 - (scale_factor - 1) * w / 2))
                y1 = max(0, int(y1 - (scale_factor - 1) * h / 2))
                x2 = min(frame.shape[1], int(x2 + (scale_factor - 1) * w / 2))
                y2 = min(frame.shape[0], int(y2 + (scale_factor - 1) * h / 2))
                
                roi = processed_frame[y1:y2, x1:x2]
                # Mélange des pixels de la boîte (par exemple)
                temp = cv2.resize(roi, (4, 4), interpolation=cv2.INTER_LINEAR)
                pixel_roi = cv2.resize(temp, (x2 - x1, y2 - y1), interpolation=cv2.INTER_NEAREST) 
    
                processed_frame[(y1):(y2), (x1):(x2)] = pixel_roi
            else :
                for box in results[0].boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    w, h = x2 - x1, y2 - y1
                    x1 = max(0, int(x1 - (scale_factor - 1) * w / 2))
                    y1 = max(0, int(y1 - (scale_factor - 1) * h / 2))
                    x2 = min(frame.shape[1], int(x2 + (scale_factor - 1) * w / 2))
                    y2 = min(frame.shape[0], int(y2 + (scale_factor - 1) * h / 2))
                    
                    roi = processed_frame[y1:y2, x1:x2]
                    # Mélange des pixels de la boîte (par exemple)
                    temp = cv2.resize(roi, (4, 4), interpolation=cv2.INTER_LINEAR)
                    pixel_roi = cv2.resize(temp, (x2 - x1, y2 - y1), interpolation=cv2.INTER_NEAREST) 
            
                    processed_frame[(y1):(y2), (x1):(x2)] = pixel_roi
            

        show_frame(processed_frame)
        root.after(30, update_frameP)
        output_writer.write(processed_frame)


    update_frameP()


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

btn_procS_video = tk.Button(root, text="Traiter une vidéo (mélange)", command=shuffle_video)
btn_procS_video.pack()

btn_procB_video = tk.Button(root, text="Traiter une vidéo (flou)", command=blur_video)
btn_procB_video.pack()

btn_procP_video = tk.Button(root, text="Traiter une vidéo (pixel)", command=pixel_video)
btn_procP_video.pack()

user_Entry = Entry(root,bg="white")
user = Label(root, text = "Numéro de la boîte")
user.pack()
user_Entry.pack() 

c1 = tk.Checkbutton(root,
                text='Poursuite de cible',
                variable=track,
                onvalue= 1,
                offvalue= 0)
c1.pack()

# btn_save_video = tk.Button(root, text="Sauvegarder la vidéo", command=save_video)
# btn_save_video.pack()

label_video = tk.Label(root)
label_video.pack()

# Lancer l'application

root.mainloop()
    
