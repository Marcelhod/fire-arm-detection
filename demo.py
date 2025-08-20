import cv2
import os
from ultralytics import YOLO
import numpy as np

# Replace with your camera's RTSP URL
RTSP_URL = "rtsp://admin:[password]@[IP]:[port]/cam/realmonitor?channel=1&subtype=0"

# Load your custom trained model
path = r'C:\Users\Srour\Downloads\best (2).pt'
model = YOLO(path)
model.model.names = {0: 'knife', 1: 'gun', 2: 'shotgun', 3: 'riffle'}

cap = cv2.VideoCapture(RTSP_URL)

if not cap.isOpened():
    print("Error: Could not open RTSP stream.")
    exit()

fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Define colors for different classes
colors = {
    'knife': (0, 0, 255),      
    'gun': (0, 165, 255),      
    'shotgun': (0, 255, 255),  
    'riffle': (0, 255, 0)      
}

# Function to process detections and draw bounding boxes
def process_detections(frame, results):
    for result in results:
        for box in result.boxes:
            # Extract box coordinates
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            conf = float(box.conf[0])
            cls_id = int(box.cls[0])
            label = model.names[cls_id]
            
            color = colors.get(label, (255, 255, 255))  
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
            
            label_text = f"{label} {conf:.2f}"
            
            (text_width, text_height), baseline = cv2.getTextSize(
                label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2
            )
            
            cv2.rectangle(
                frame, 
                (int(x1), int(y1) - text_height - baseline), 
                (int(x1) + text_width, int(y1)), 
                color, 
                -1
            )
            
            # Put text
            cv2.putText(
                frame, 
                label_text, 
                (int(x1), int(y1) - baseline), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.5, 
                (0, 0, 0),  # Black text
                2
            )
    
    return frame

# Main loop
while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to grab frame.")
        break

    results = model.predict(
        source=frame,   # Use the current frame
        conf=0.5,       # Confidence threshold
        verbose=False,  # Don't print results to console
        device='cpu'    # Use 'cuda' if you have GPU
    )
    
    frame_with_detections = process_detections(frame, results)
    
    cv2.imshow("Weapon Detection - RTSP Stream", frame_with_detections)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
