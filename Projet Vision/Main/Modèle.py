
from ultralytics import YOLO
model = YOLO("yolo26n.pt")
metrics = model.val(data="dataset.yaml")
print(f"Précision globale : {metrics.box.map * 100:.2f} %")
