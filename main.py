# Load a model
from ultralytics import YOLO

model = YOLO("yolov8n.yaml")  # build a new model from scratch

# Use the model
model.train(data="resources/data.yaml", epochs=2)  # train the model
metrics = model.val()  # evaluate model performance on the validation set
