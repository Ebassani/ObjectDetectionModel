import torch
from ultralytics import YOLO


class ObjectDetection:
    def __init__(self):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.model = self.load_model()

    def load_model(self):
        model = YOLO("runs/detect/train28/weights/best.pt")
        model.fuse()

        return model

    def predict(self, image):
        results = self.model(image)

        return results

    def get_boxes(self, results):

        for result in results:
            boxes = result.boxes.cpu().numpy()

            print(boxes)
            print("======================")
            print(boxes.xyxy[5])
