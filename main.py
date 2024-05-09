from ultralytics import YOLO
import segmentation

# model = YOLO("runs/detect/train30/weights/best.pt")

# model.train(data="resources/Ingredient detection.v1i.yolov8/data.yaml", epochs=120)  # train the model
# metrics = model.val()  # evaluate model performance on the validation set

# model.export(format="onnx")

# print("================")


object_detection = segmentation.ObjectDetection("runs/detect/train30/weights/best.pt")

labels = object_detection.get_labels("resources/images/cebola-stuttgarter.jpg", 0.6)

print(labels)
