import tensorflow as tf


def yolo_loss(y_true, y_pred):
    squared_error = tf.math.squaredDifference(y_pred, y_true)

    confidence_loss = tf.nn.softmaxCrossEntropyWithLogits(y_pred, y_true).loss()

    mean_squared_error = tf.reduce_mean(squared_error)

    mean_confidence_loss = tf.reduce_mean(confidence_loss)

    total_loss = tf.math.add(mean_squared_error, mean_confidence_loss)

    return total_loss
