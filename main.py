from losses import yolo_loss
from models import create_yolo_model


def compile_model():
    shape = (448, 448, 3)
    num_classes = 20

    model = create_yolo_model(shape, num_classes)

    model.compile(optimizer='adam', loss=yolo_loss)


if __name__ == '__main__':
    compile_model()
