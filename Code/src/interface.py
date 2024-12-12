import tkinter as tk
from tkinter import *
from tkinter import filedialog
from tkinter import ttk
import cv2
from ultralytics import YOLO
import numpy as np
from PIL import Image, ImageTk
import json

root = tk.Tk()
root.title("Détection et traitement avec YOLO")

# Initialiser YOLO
model = YOLO("yolov10n-face.pt")  # Chargez un modèle YOLO compatible

# Variables globales
video_capture = None
output_writer = None
scale_factor = 1.0
random_seed = 4
frame = None
processed_frame = None
track = tk.IntVar()
trackers = []
tracker_histograms = []
frame_counter=0
time_frame_replace = tk.IntVar()

def calculate_histogram(image, bbox):
    x1, y1, w, h = map(int, bbox)
    roi = image[y1:y1+h, x1:x1+w]
    hist = cv2.calcHist([roi], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
    cv2.normalize(hist, hist)
    return hist

def compare_histograms(hist1, hist2):
    return cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)

def reset_tracker(processed_frame, tracker_index, original_hist):
    global model, trackers, tracker_histograms
    results = model(processed_frame)
    best_similarity = 0
    best_box = None

    # Comparer chaque boîte détectée avec l'apparence originale
    for box in results[0].boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        bbox = (x1, y1, x2 - x1, y2 - y1)
        hist = calculate_histogram(processed_frame, bbox)
        similarity = compare_histograms(original_hist, hist)
        print(f"similarity reset = {similarity}")
        if similarity > best_similarity:
            best_similarity = similarity
            best_box = bbox

    # Si une correspondance suffisante est trouvée, réinitialiser le tracker
    if best_box and best_similarity > 0.7:  # Seuil de similarité ajustable
        print(f"Réinitialisation du tracker {tracker_index}")
        trackers[tracker_index] = cv2.TrackerCSRT_create()
        trackers[tracker_index].init(processed_frame, best_box)
        tracker_histograms[tracker_index] = calculate_histogram(processed_frame, best_box)
    else:
        print(f"Impossible de réinitialiser le tracker {tracker_index} : cible introuvable")

# Charger une vidéo
def load_video():
    global video_capture, frame, processed_frame,frame_counter
    filepath = filedialog.askopenfilename(defaultextension=".mp4",
                                          filetypes=[("MP4 files", "*.mp4"),
                                                     ("AVI files", "*.avi")])
    if filepath:
        video_capture = cv2.VideoCapture(filepath)
        ret, frame = video_capture.read()
        processed_frame = frame.copy()
        results = model(frame)
        i = 0

        frame_counter=0
        tracker_histograms.clear()
        trackers.clear()

        for box in results[0].boxes:
            i = i+1
            x1, y1, x2, y2 = map(int, box.xyxy[0])  # Obtenir les coordonnées de la boîte
            #boxes.append([x1, y1, x2, y2])

            # Créer un tracker pour chaque visage détecté
            tracker = cv2.TrackerCSRT_create()

            trackers.append(tracker)
            tracker.init(frame, (x1, y1, x2 - x1, y2 - y1)) 
            tracker_histograms.append(calculate_histogram(frame, (x1, y1, x2 - x1, y2 - y1)))

            # Dessiner la boîte sur l'image
            cv2.rectangle(processed_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Dessiner en vert
            cv2.putText(processed_frame, f"{i}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        show_frame(processed_frame)

# Traiter une vidéo
def shuffle_video():
    global video_capture, output_writer, processed_frame, video_boxes,frame_counter
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
    scale_factor = int(scale_Entry.get())

    # Initialiser VideoWriter
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    output_writer = cv2.VideoWriter(output_filepath, fourcc, fps, (frame_width, frame_height))

    video_capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
    video_boxes = []

    def update_frameS():
        global frame, processed_frame,frame_counter
        ret, frame = video_capture.read()
        
        
        
        if not ret:
            video_capture.release()
            output_writer.release()
            print("Processing complete.")

            return

        processed_frame = frame.copy()
        frame_counter+=1 
        frame_boxes = []

        if track.get() == 1 :        
            if int(user_Entry.get()) > 0:
                success, bbox = trackers[int(user_Entry.get())-1].update(processed_frame)
                if success:
                    x1, y1, w, h = map(int, bbox)
                    x2, y2 = x1 + w, y1 + h
                    frame_boxes.append({"x1": x1, "y1": y1, "x2": x2, "y2": y2})

                    if frame_counter % int(rafraiche_Entry.get()) == 0:
                        tracker_histograms[int(user_Entry.get())-1] = calculate_histogram(processed_frame, (x1, y1, w, h))

                    hist = calculate_histogram(processed_frame, bbox)
                    similarity = compare_histograms(tracker_histograms[int(user_Entry.get())-1], hist)

                    # Seuil de similarité pour considérer la région comme valide
                    print(f"similarity = {similarity}")
                    if similarity < float(seuil_Entry.get()):
                        print(f"Tracker {int(user_Entry.get())-1}  a perdu la cible, tentative de réinitialisation...")
                        reset_tracker(processed_frame, int(user_Entry.get())-1, tracker_histograms[int(user_Entry.get())-1])
                        # Option : marquer la boîte ou réinitialiser le tracker
                    else:

                        # Extraire la région d'intérêt
                        roi = processed_frame[y1:y2, x1:x2]

                        # Vérification du type
                        assert processed_frame.dtype == np.uint8, "processed_frame doit être en uint8."

                        # Mélanger les pixels dans la région
                        pixels = roi.reshape(-1, 3)
                        seed = 42  # Définir une graine
                        rng = np.random.default_rng(seed)

                        # Générer et appliquer la permutation
                        permutation = rng.permutation(len(pixels))
                        pixels_mixed = pixels[permutation]

                        # Remettre les pixels mélangés dans l'image
                        roi_mixed = pixels_mixed.reshape(roi.shape)

                        # Clip pour s'assurer que les valeurs sont dans les limites
                        roi_mixed = np.clip(roi_mixed, 0, 255).astype(np.uint8)

                        # Réinsérer dans l'image originale
                        processed_frame[y1:y2, x1:x2] = roi_mixed

                        # Validation des dimensions
                        assert processed_frame[y1:y2, x1:x2].shape == roi.shape, "Erreur de forme après réinsertion."

                    video_boxes.append({"frame_counter": frame_counter, "boxes": frame_boxes})         
            else:
                for i, tracker in enumerate(trackers):
                    success, bbox = tracker.update(processed_frame)
                    if success:
                        x1, y1, w, h = map(int, bbox)
                        x2, y2 = x1 + w, y1 + h
                        frame_boxes.append({"x1": x1, "y1": y1, "x2": x2, "y2": y2})

                        if frame_counter % int(rafraiche_Entry.get()) == 0:
                            tracker_histograms[i] = calculate_histogram(processed_frame, (x1, y1, w, h))

                        hist = calculate_histogram(processed_frame, bbox)
                        similarity = compare_histograms(tracker_histograms[i], hist)

                        # Seuil de similarité pour considérer la région comme valide
                        print(f"similarity = {similarity}")
                        if similarity < float(seuil_Entry.get()):
                            print(f"Tracker {i}  a perdu la cible, tentative de réinitialisation...")
                            reset_tracker(processed_frame, i, tracker_histograms[i])
                            # Option : marquer la boîte ou réinitialiser le tracker
                        else:
                            roi = processed_frame[y1:y2, x1:x2]

                            # Mélanger les pixels dans la boîte détectée
                            pixels = roi.reshape(-1, 3)
                            seed = 42  # Définir une graine
                            rng = np.random.default_rng(seed)
                            permutation = rng.permutation(len(pixels))  # Générer une permutation
                            pixels_mixed = pixels[permutation]          # Mélanger les pixels
                            processed_frame[y1:y2, x1:x2] = pixels_mixed.reshape(roi.shape)
                        video_boxes.append({"frame_counter": frame_counter, "boxes": frame_boxes}) 
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
                frame_boxes.append({"x1": x1, "y1": y1, "x2": x2, "y2": y2})
                
                roi = processed_frame[y1:y2, x1:x2]
                # Mélange des pixels de la boîte (par exemple)
                pixels = roi.reshape(-1, 3)
                seed = 42  # Définir une graine
                rng = np.random.default_rng(seed)
                permutation = rng.permutation(len(pixels))  # Générer une permutation
                pixels_mixed = pixels[permutation]          # Mélanger les pixels
                processed_frame[y1:y2, x1:x2] = pixels_mixed.reshape(roi.shape) 
                video_boxes.append({"frame_counter": frame_counter, "boxes": frame_boxes}) 
            else :
                for box in results[0].boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    w, h = x2 - x1, y2 - y1
                    x1 = max(0, int(x1 - (scale_factor - 1) * w / 2))
                    y1 = max(0, int(y1 - (scale_factor - 1) * h / 2))
                    x2 = min(frame.shape[1], int(x2 + (scale_factor - 1) * w / 2))
                    y2 = min(frame.shape[0], int(y2 + (scale_factor - 1) * h / 2))
                    
                    frame_boxes.append({"x1": x1, "y1": y1, "x2": x2, "y2": y2})


                    roi = processed_frame[y1:y2, x1:x2]
                    # Mélange des pixels de la boîte (par exemple)
                    pixels = roi.reshape(-1, 3)
                    seed = 42  # Définir une graine
                    rng = np.random.default_rng(seed)
                    permutation = rng.permutation(len(pixels))  # Générer une permutation
                    pixels_mixed = pixels[permutation]          # Mélanger les pixels
                    processed_frame[y1:y2, x1:x2] = pixels_mixed.reshape(roi.shape) 
                video_boxes.append({"frame_counter": frame_counter, "boxes": frame_boxes})      
        show_frame(processed_frame)
        root.after(30, update_frameS)
        output_writer.write(processed_frame)
        with open(str(fichier_Entry.get()), "w") as json_file:
            json.dump(video_boxes, json_file, indent=4)

        #print("Coordonnées des boîtes pour toute la vidéo écrites dans 'video_boxes_coordinates.json'")
    
    

    update_frameS()
    

def blur_video():
    global video_capture, output_writer, processed_frame, frame_counter
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
    scale_factor = int(scale_Entry.get())

    # Initialiser VideoWriter
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    output_writer = cv2.VideoWriter(output_filepath, fourcc, fps, (frame_width, frame_height))

    video_capture.set(cv2.CAP_PROP_POS_FRAMES, 0)

    def update_frameB():
        global frame, processed_frame,frame_counter
        ret, frame = video_capture.read()
        
        if not ret:
            video_capture.release()
            output_writer.release()
            print("Processing complete.")

            return

        processed_frame = frame.copy()
        frame_counter+=1 

        if track.get() == 1 :        
            if int(user_Entry.get()) > 0:
                success, bbox = trackers[int(user_Entry.get())-1].update(processed_frame)
                if success:
                    x1, y1, w, h = map(int, bbox)
                    x2, y2 = x1 + w, y1 + h

                    if frame_counter % int(rafraiche_Entry.get()) == 0:
                        tracker_histograms[int(user_Entry.get())-1] = calculate_histogram(processed_frame, (x1, y1, w, h))

                    hist = calculate_histogram(processed_frame, bbox)
                    similarity = compare_histograms(tracker_histograms[int(user_Entry.get())-1], hist)

                    # Seuil de similarité pour considérer la région comme valide
                    print(f"similarity = {similarity}")
                    if similarity < float(seuil_Entry.get()):
                        print(f"Tracker {int(user_Entry.get())-1}  a perdu la cible, tentative de réinitialisation...")
                        reset_tracker(processed_frame, int(user_Entry.get())-1, tracker_histograms[int(user_Entry.get())-1])
                        # Option : marquer la boîte ou réinitialiser le tracker

                    else :
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

                        if frame_counter % int(rafraiche_Entry.get()) == 0:
                            tracker_histograms[i] = calculate_histogram(processed_frame, (x1, y1, w, h))

                        hist = calculate_histogram(processed_frame, bbox)
                        similarity = compare_histograms(tracker_histograms[i], hist)

                        # Seuil de similarité pour considérer la région comme valide
                        print(f"similarity = {similarity}")
                        if similarity < float(seuil_Entry.get()):
                            print(f"Tracker {i}  a perdu la cible, tentative de réinitialisation...")
                            reset_tracker(processed_frame, i, tracker_histograms[i])
                            # Option : marquer la boîte ou réinitialiser le tracker
                        else :

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
    
    global video_capture, output_writer, processed_frame,frame_counter
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
    scale_factor = int(scale_Entry.get())

    # Initialiser VideoWriter
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    output_writer = cv2.VideoWriter(output_filepath, fourcc, fps, (frame_width, frame_height))

    video_capture.set(cv2.CAP_PROP_POS_FRAMES, 0)

    def update_frameP():
        global frame, processed_frame,frame_counter
        ret, frame = video_capture.read()
        if not ret:
            video_capture.release()
            output_writer.release()
            return
        
        processed_frame = frame.copy()
        frame_counter+=1 
        if track.get() == 1 :        
            if int(user_Entry.get()) > 0:
                success, bbox = trackers[int(user_Entry.get())-1].update(processed_frame)
                if success:
                    x1, y1, w, h = map(int, bbox)
                    x2, y2 = x1 + w, y1 + h

                    if frame_counter % int(rafraiche_Entry.get()) == 0:
                        tracker_histograms[int(user_Entry.get())-1] = calculate_histogram(processed_frame, (x1, y1, w, h))

                    hist = calculate_histogram(processed_frame, bbox)
                    similarity = compare_histograms(tracker_histograms[int(user_Entry.get())-1], hist)

                    # Seuil de similarité pour considérer la région comme valide
                    print(f"similarity = {similarity}")
                    if similarity < float(seuil_Entry.get()):
                        print(f"Tracker {int(user_Entry.get())-1}  a perdu la cible, tentative de réinitialisation...")
                        reset_tracker(processed_frame, int(user_Entry.get())-1, tracker_histograms[int(user_Entry.get())-1])
                        # Option : marquer la boîte ou réinitialiser le tracker

                    else :
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

                        if frame_counter % int(rafraiche_Entry.get()) == 0:
                            tracker_histograms[i] = calculate_histogram(processed_frame, (x1, y1, w, h))

                        hist = calculate_histogram(processed_frame, bbox)
                        similarity = compare_histograms(tracker_histograms[i], hist)

                        # Seuil de similarité pour considérer la région comme valide
                        print(f"similarity = {similarity}")
                        if similarity < float(seuil_Entry.get()):
                            print(f"Tracker {i}  a perdu la cible, tentative de réinitialisation...")
                            reset_tracker(processed_frame, i, tracker_histograms[i])
                            # Option : marquer la boîte ou réinitialiser le tracker
                        else :
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

def dechiffre():
    global video_capture, output_writer, processed_frame, video_boxes, frame_counter, data
    if not video_capture:
        return

    output_filepath = filedialog.asksaveasfilename(defaultextension=".mp4",
                                                   filetypes=[("MP4 files", "*.mp4"),
                                                              ("AVI files", "*.avi")])
    if not output_filepath:
        return


    dechiffre_file = filedialog.asksaveasfilename(defaultextension=".json",
                                                   filetypes=[("json files", "*.json")])
    if not dechiffre_file:
        return
    # Obtenir les propriétés de la vidéo
    frame_width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(video_capture.get(cv2.CAP_PROP_FPS))

    # Initialiser VideoWriter
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    output_writer = cv2.VideoWriter(output_filepath, fourcc, fps, (frame_width, frame_height))

    video_capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
    video_boxes = []
    frame_counter = 0

    with open(dechiffre_file, "r") as json_file:
        data = json.load(json_file)

    def get_boxes_for_frame(frame_counter, data):
        for frame in data:
            if frame["frame_counter"] == frame_counter:
                return frame["boxes"]  # Retourne la liste des boîtes
        return []  # Si aucune boîte n'est trouvée pour la frame

    def update_frameD():
        global frame, processed_frame, frame_counter, data
        ret, frame = video_capture.read()
        
        
        
        if not ret:
            video_capture.release()
            output_writer.release()
            print("Processing complete.")

            return

        processed_frame = frame.copy()

        frame_counter += 1 
        boxes = get_boxes_for_frame(frame_counter, data)
        for box in boxes:
            x1, y1, x2, y2 = box["x1"], box["y1"], box["x2"], box["y2"]

            # Extraire la région d'intérêt
            roi = processed_frame[y1:y2, x1:x2]

            # Vérification du type
            assert processed_frame.dtype == np.uint8, "L'image processed_frame doit être en uint8."

            # Mélanger les pixels de la boîte
            pixels_mixed = roi.reshape(-1, 3)

            # Définir une graine
            seed = 42
            num_pixels = len(pixels_mixed)

            # Recréer la permutation originale avec la graine
            rng = np.random.default_rng(seed)
            recreated_permutation = rng.permutation(num_pixels)  # Recrée la permutation

            # Inverser la permutation
            inverse_permutation = np.argsort(recreated_permutation)

            # Restaurer les pixels à l'ordre original
            pixels_restored = pixels_mixed[inverse_permutation]
            roi_restored = pixels_restored.reshape(roi.shape)

            # Vérification des dimensions
            assert roi_restored.shape == roi.shape, "Les dimensions de la région restaurée ne correspondent pas à celles d'origine."

            # Clip pour garantir les valeurs valides
            roi_restored = np.clip(roi_restored, 0, 255).astype(np.uint8)

            # Réinsérer dans l'image d'origine
            processed_frame[y1:y2, x1:x2] = roi_restored   
        show_frame(processed_frame)
        root.after(30, update_frameD)
        output_writer.write(processed_frame)
    

    update_frameD()


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

frame0 = tk.Frame(root)
frame0.pack(pady=5)  # Ajout d'un espacement vertical

btn_procB_video = tk.Button(frame0, text="Traiter une vidéo (flou)", command=blur_video)
btn_procB_video.pack()

btn_procP_video = tk.Button(frame0, text="Traiter une vidéo (pixel)", command=pixel_video)
btn_procP_video.pack()

frame1 = tk.Frame(root)
frame1.pack(pady=5)  # Ajout d'un espacement vertical

btn_procS_video = tk.Button(frame1, text="Traiter une vidéo (mélange)", command=shuffle_video)
btn_procS_video.pack()

btn_procD_video = tk.Button(frame1, text="Déchiffrer un mélange", command=dechiffre)
btn_procD_video.pack()



frame2 = tk.Frame(root)
frame2.pack(pady=5)

user = tk.Label(frame2, text="Numéro boîte")
user.pack(side=tk.LEFT, padx=5)  # Espacement horizontal
user_Entry = tk.Entry(frame2, bg="white", width=5)  # Définir la largeur
user_Entry.pack(side=tk.LEFT)

# Deuxième ligne
rafraiche = tk.Label(frame2, text="Taux vérif")
rafraiche.pack(side=tk.LEFT, padx=5)
rafraiche_Entry = tk.Entry(frame2, bg="white", width=5)
rafraiche_Entry.pack(side=tk.LEFT)

# Troisième ligne
frame3 = tk.Frame(root)
frame3.pack(pady=5)
seuil = tk.Label(frame3, text="Seuil")
seuil.pack(side=tk.LEFT, padx=5)
seuil_Entry = tk.Entry(frame3, bg="white", width=5)
seuil_Entry.pack(side=tk.LEFT)

scale = tk.Label(frame3, text="Taille boîte")
scale.pack(side=tk.LEFT, padx=5)
scale_Entry = tk.Entry(frame3, bg="white", width=5)
scale_Entry.pack(side=tk.LEFT)

c1 = tk.Checkbutton(frame3,
                text='Poursuite de cible',
                variable=track,
                onvalue= 1,
                offvalue= 0)
c1.pack()

frame4 = tk.Frame(root)
frame4.pack(pady=5)
fichier = tk.Label(frame4, text="Fichier JSON")
fichier.pack(side=tk.LEFT, padx=5)
fichier_Entry = tk.Entry(frame4, bg="white", width=10)
fichier_Entry.pack(side=tk.LEFT)

# btn_save_video = tk.Button(root, text="Sauvegarder la vidéo", command=save_video)
# btn_save_video.pack()

label_video = tk.Label(root)
label_video.pack()

# Lancer l'application

root.mainloop()
    
