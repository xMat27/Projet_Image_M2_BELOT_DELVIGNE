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
output_path = "video_output.mp4"
fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # Codec vidéo
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

# Traitement de chaque frame
while cap.isOpened():
    ret, frame = cap.read()  # Lire une frame de la vidéo
    if not ret:
        break  # Fin de la vidéo

    # Exécuter YOLO sur la frame
    results = model(frame)  # liste d'objets Results

    # Dessiner les boîtes englobantes sur la frame
    for box in results[0].boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])  # Extraire les coordonnées de la boîte
        confidence = box.conf[0]  # Confiance de la détection

        # Dessiner la boîte englobante et ajouter la confiance
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"{confidence:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Écrire la frame annotée dans la vidéo de sortie
    out.write(frame)

    
