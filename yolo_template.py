from ultralytics import YOLO
import shutil

model = YOLO('yolov8n.pt')

name = "myYolov8n"

model.train(
    data='/kaggle/input/chessboard-cv/data.yaml',
    epochs=100,
    imgsz=940,
    batch=64,
    freeze=10,
    name=name,
    device=0,
)

shutil.copy('runs/detect/' + name + '/weights/best.pt', '/kaggle/working/best.pt')