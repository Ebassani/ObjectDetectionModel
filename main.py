# Load a model
from ultralytics import YOLO
import segmentation

# model = YOLO("runs/detect/train28/weights/best.pt")

# Use the model
# model.train(data="resources/Ingredient detection.v1i.yolov8/data.yaml", epochs=15)  # train the model
# metrics = model.val()  # evaluate model performance on the validation set

# model.export(format="onnx")

# print("================")

# results = model("resources/images/cebola-stuttgarter.jpg")
#
#
# for result in results:
#     print("==========")
#     print(result)

object_detection = segmentation.ObjectDetection()

results = object_detection.predict("resources/images/cebola-stuttgarter.jpg")

object_detection.get_boxes(results)