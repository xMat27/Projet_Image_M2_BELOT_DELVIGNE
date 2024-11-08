import cv2
from ultralytics import YOLO

# Charger le modèle YOLO
model = YOLO("yolov10n-face.pt")  # Utilisez un modèle compatible

# Charger la vidéo d'entrée
input_path = "videos/Ronaldo_cut.avi"
cap = cv2.VideoCapture(input_path)

# Vérifier si la vidéo s'ouvre correctement
if not cap.isOpened():
    print("Erreur lors de l'ouverture de la vidéo")
    exit()

# Récupérer les dimensions de la vidéo et le nombre d'images par seconde (fps)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# Configurer la vidéo de sortie avec VideoWriter
output_path = "video_flou.mp4"
fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # Codec vidéo
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
scale_factor = 1.3

# Traitement de chaque frame
while cap.isOpened():
    ret, frame = cap.read()  # Lire une frame de la vidéo
    if not ret:
        break  # Fin de la vidéo

    # Exécuter YOLO sur la frame
    results = model(frame)  # liste d'objets Results

    # Dessiner les boîtes englobantes sur la frame
    for box in results[0].boxes:
         # Extraire les coordonnées et agrandir la boîte
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        w, h = x2 - x1, y2 - y1

        # Calculer les nouvelles dimensions
        x1 = max(0, int(x1 - (scale_factor - 1) * w / 2))
        y1 = max(0, int(y1 - (scale_factor - 1) * h / 2))
        x2 = min(width, int(x2 + (scale_factor - 1) * w / 2))
        y2 = min(height, int(y2 + (scale_factor - 1) * h / 2))


        # Extraire la région d'intérêt (ROI) pour flouter
        roi = frame[(y1):(y2), (x1):(x2)]

        # Appliquer le flou sur cette région
        blurred_roi = cv2.GaussianBlur(roi, (51, 51), 0)  # Utiliser un flou gaussien
        #temp = cv2.resize(roi, (4, 4), interpolation=cv2.INTER_LINEAR)
        #blurred_roi = cv2.resize(temp, (x2 - x1, y2 - y1), interpolation=cv2.INTER_NEAREST)

        # Remettre la zone floutée dans l'image
        frame[(y1):(y2), (x1):(x2)] = blurred_roi

    # Écrire la frame annotée dans la vidéo de sortie
    out.write(frame)

    
