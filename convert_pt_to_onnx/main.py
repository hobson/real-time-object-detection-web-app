from ultralytics import YOLO

# Load the YOLO model
model = YOLO("yolo12n.pt")

# Export the model to ONNX format - static shape + baked-in NMS, see
# ultralytics_pt_to_onnx.md's "Preferred export" section for why.
model.export(format="onnx", simplify=True, dynamic=False, imgsz=256, batch=1, nms=True, opset=19)
