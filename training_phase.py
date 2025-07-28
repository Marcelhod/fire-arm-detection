from google.colab import drive
import os
from ultralytics import YOLO
drive.mount('/content/archive')
model = YOLO("yolo11n.pt")
!yolo train model=yolo11n.pt data=/content/archive/MyDrive/FYP/google_colab_config.yaml epochs=10 imgsz=640
!mkdir /content/v11epo40
!cp /content/runs/detect/train2/weights/best.pt /content/v11epo40/v11epo40.pt
!cp -r /content/runs/detect/train2 /content/v11epo40/

%cd v11epo40
!zip /content/v11epo40.zip v11epo40.pt
!zip -r /content/v11epo40.zip train2
%cd /content
