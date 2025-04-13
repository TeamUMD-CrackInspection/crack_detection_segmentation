from ultralytics import YOLO

# Load a model
model = YOLO("checkpoints/yolov11n.pt")  # load a pretrained model (recommended for training)
# data_path = "/home/umd/workspace/crack_detection/data/custom/pattern_crack_samples/ultralytics_yolo_detection1.0/data.yaml"
data_path = "/home/umd/workspace/crack_detection/data/custom/pattern_cracks_250313/cvat_annotation/split/data.yaml"

# Train the model
results = model.train(
    data=data_path, 
    epochs=100, 
    imgsz=512, 
    roject="runs/detect/", name="pattern_crack_0313_train/ep100")

# results = model.train(
#     data=data_path, 
#     epochs=150, 
#     imgsz=512, 
#     lr0=0.001,
#     cos_lr=True,
#     optimizer="AdamW",
#     mosaic=0.8,
#     mixup=0.2,    
#     # batch=16,
#     project="runs/detect/", name="pattern_crack_0313_train/ep150")



