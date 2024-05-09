# from ultralytics import YOLO
import segmentation
from util import count_labels
import gpt_connections

# model = YOLO("runs/detect/train30/weights/best.pt")

# model.train(data="resources/Ingredient detection.v1i.yolov8/data.yaml", epochs=120)  # train the model
# metrics = model.val()  # evaluate model performance on the validation set

# model.export(format="onnx")

# print("================")


object_detection = segmentation.ObjectDetection("runs/detect/train30/weights/best.pt")

labels = object_detection.get_labels("resources/images/cebola-stuttgarter.jpg", 0.6)

counted_labels = count_labels(labels)
# print(counted_labels)

gpt_assistant = gpt_connections.GPTAssistant()
gpt_assistant.ask_for_recipe(counted_labels)
