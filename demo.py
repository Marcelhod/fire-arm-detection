from IPython.display import display, HTML, clear_output
import ipywidgets as widgets
from google.colab import files
import cv2
import numpy as np
from ultralytics import YOLO
import matplotlib.pyplot as plt
import time

model = YOLO('/content/runs/detect/train4/weights/best.pt')  
model.overrides['conf'] = 0.5  
model.overrides['iou'] = 0.45  
model.overrides['agnostic_nms'] = True  
model.overrides['max_det'] = 1000  

WEAPON_COLOR = (255, 0, 0)  
TEXT_COLOR = (255, 255, 255)  
BOX_THICKNESS = 2
FONT_SCALE = 0.7
FONT_THICKNESS = 2


upload_btn = widgets.FileUpload(
    accept='image/*',
    multiple=False,
    description='Upload Image'
)

detect_btn = widgets.Button(
    description='Detect Weapons',
    button_style='success'
)

confidence_slider = widgets.FloatSlider(
    value=0.5,
    min=0.1,
    max=0.9,
    step=0.05,
    description='Confidence:'
)

performance_switch = widgets.ToggleButton(
    value=False,
    description='Show Performance Stats',
    tooltip='Toggle performance metrics'
)

output = widgets.Output()

def optimized_draw_boxes(image, boxes):
    """Vectorized drawing for better performance"""
    img = image.copy()
    for box in boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
        conf = float(box.conf)
        
        
        cv2.rectangle(img, (x1, y1), (x2, y2), WEAPON_COLOR, BOX_THICKNESS)
        
        
        label = f"Weapon {conf:.2f}"
        (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, FONT_THICKNESS)
        
        
        cv2.rectangle(img, (x1, y1 - h - 10), (x1 + w, y1), WEAPON_COLOR, -1)
        
        
        cv2.putText(img, label, (x1, y1 - 5), 
                   cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, TEXT_COLOR, FONT_THICKNESS)
    return img

def on_detect_click(b):
    with output:
        clear_output()
        
        if not upload_btn.value:
            print("⚠️ Please upload an image first")
            return
            
        try:
            
            start_time = time.time()
            uploaded_file = next(iter(upload_btn.value.values()))
            img = cv2.imdecode(np.frombuffer(uploaded_file['content'], np.uint8), cv2.IMREAD_COLOR)
            load_time = time.time() - start_time
            
            
            start_time = time.time()
            results = model.predict(
                img,
                conf=confidence_slider.value,
                imgsz=640,
                augment=False,  
                verbose=False  
            )
            detect_time = time.time() - start_time
            
            start_time = time.time()
            annotated_img = optimized_draw_boxes(img, results[0].boxes)
            process_time = time.time() - start_time
            
            display_img = cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB)
            
            plt.figure(figsize=(12, 8))
            plt.imshow(display_img)
            plt.axis('off')
            plt.show()
            
            weapon_count = len(results[0].boxes)
            print("🔴 Weapon Detection Results:")
            print("-" * 40)
            print(f"Detected weapons: {weapon_count}")
            for i, box in enumerate(results[0].boxes, 1):
                print(f"{i}. Confidence: {float(box.conf):.2f}")
            
            if weapon_count == 0:
                print("No weapons detected (try lowering confidence threshold)")
            
            if performance_switch.value:
                print("\n⚡ Performance Metrics:")
                print(f"- Image load time: {load_time:.3f}s")
                print(f"- Detection time: {detect_time:.3f}s")
                print(f"- Drawing time: {process_time:.3f}s")
                print(f"- Total processing time: {load_time+detect_time+process_time:.3f}s")
                print(f"- FPS: {1/(detect_time+process_time):.1f}")
            
        except Exception as e:
            print(f"❌ Error: {str(e)}")

detect_btn.on_click(on_detect_click)

display(HTML("""
<h2 style='color: #d22;'>Optimized Weapon Detection</h2>
<p>Upload an image to detect weapons (highlighted in <span style='color:red;'>red</span>)</p>
"""))
display(widgets.VBox([
    confidence_slider,
    performance_switch,
    upload_btn, 
    detect_btn
]))
display(output)
