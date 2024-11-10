import cv2
from ultralytics import YOLO


model = YOLO("yolov10n-face.pt")   


input_path = "videos/Ronaldo_cut.avi"
cap = cv2.VideoCapture(input_path)


if not cap.isOpened():
    print("Erreur lors de l'ouverture de la vidéo")
    exit()


width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))


output_path = "video_blur.mp4"
fourcc = cv2.VideoWriter_fourcc(*"mp4v")  
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
scale_factor = 1.3

while cap.isOpened():
    ret, frame = cap.read()  
    if not ret:
        break  


    results = model(frame) 


    for box in results[0].boxes:
        
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        w, h = x2 - x1, y2 - y1

        # Agrandissement de la boîte
        x1 = max(0, int(x1 - (scale_factor - 1) * w / 2))
        y1 = max(0, int(y1 - (scale_factor - 1) * h / 2))
        x2 = min(width, int(x2 + (scale_factor - 1) * w / 2))
        y2 = min(height, int(y2 + (scale_factor - 1) * h / 2))


        # Region Of Interest
        roi = frame[(y1):(y2), (x1):(x2)]

        # Appliquer le flou 
        blurred_roi = cv2.GaussianBlur(roi, (51, 51), 0)  
    
        frame[(y1):(y2), (x1):(x2)] = blurred_roi


    out.write(frame)
    cv2.imwrite("frame_blur.jpg", frame)

    
