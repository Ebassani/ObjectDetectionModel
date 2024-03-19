import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense


def create_yolo_model(shape, num_classes):
    model = tf.keras.Sequential([
        Conv2D(64, (7, 7), strides=(2, 2), activation='relu', padding='same', input_shape=shape),
        MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same'),
        Conv2D(192, (3, 3), activation='relu', padding='same'),
        MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same'),
        Conv2D(128, (1, 1), activation='relu', padding='same'),
        Conv2D(256, (3, 3), activation='relu', padding='same'),
        Conv2D(256, (1, 1), activation='relu', padding='same'),
        Conv2D(512, (3, 3), activation='relu', padding='same'),
        MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same'),
        Conv2D(256, (1, 1), activation='relu', padding='same'),
        Conv2D(512, (3, 3), activation='relu', padding='same'),
        Conv2D(256, (1, 1), activation='relu', padding='same'),
        Conv2D(512, (3, 3), activation='relu', padding='same'),
        Conv2D(256, (1, 1), activation='relu', padding='same'),
        Conv2D(512, (3, 3), activation='relu', padding='same'),
        Conv2D(256, (1, 1), activation='relu', padding='same'),
        Conv2D(512, (3, 3), activation='relu', padding='same'),
        Conv2D(512, (1, 1), activation='relu', padding='same'),
        Conv2D(1024, (3, 3), activation='relu', padding='same'),
        MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same'),
        Conv2D(512, (1, 1), activation='relu', padding='same'),
        Conv2D(1024, (3, 3), activation='relu', padding='same'),
        Conv2D(512, (1, 1), activation='relu', padding='same'),
        Conv2D(1024, (3, 3), activation='relu', padding='same'),
        Conv2D(1024, (3, 3), activation='relu', padding='same'),
        Conv2D(1024, (3, 3), strides=(2, 2), activation='relu', padding='same'),
        Conv2D(1024, (3, 3), activation='relu', padding='same'),
        Conv2D(1024, (3, 3), activation='relu', padding='same'),
        Flatten(),
        Dense(4096, activation='relu'),
        Dense(7 * 7 * (5 + num_classes), activation='softmax')
    ])

    return model
