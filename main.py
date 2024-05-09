# Load a model
from ultralytics import YOLO
import segmentation

# model = YOLO("runs/detect/train30/weights/best.pt")

# Use the model
# model.train(data="resources/Ingredient detection.v1i.yolov8/data.yaml", epochs=120)  # train the model
# metrics = model.val()  # evaluate model performance on the validation set

# model.export(format="onnx")

# print("================")


object_detection = segmentation.ObjectDetection("runs/detect/train30/weights/best.pt")

data = object_detection.get_data("resources/images/cebola-stuttgarter.jpg")

labels = object_detection.get_labels(data, 0.6)

print(labels)
