import torch
from ultralytics import YOLO
from util import indexes_to_labels

class ObjectDetection:
    def __init__(self, path):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.path = path

        self.model = self.load_model()

    def load_model(self):
        model = YOLO(self.path)
        model.fuse()

        return model

    def predict(self, image):
        results = self.model(image)

        return results

    def get_data(self, image):

        results = self.predict(image)

        data = []

        for result in results:
            boxes = result.boxes.cpu().numpy()

            data = boxes.data

        return data

    def get_labels(self, image, min_confidence):
        indexes = []

        data = self.get_data(image)

        for box in data:
            if box[4] > min_confidence:
                indexes.append(int(box[5]))

        return indexes_to_labels(indexes, "resources/labels.txt")
