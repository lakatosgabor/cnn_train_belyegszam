from ultralytics import YOLO

model = YOLO("yolo11n-cls.pt")  # load a pretrained model (recommended for training)

results = model.train(data=r"E:\PythonProjects\ocr\datasets\mnist", epochs=130, imgsz=28)
