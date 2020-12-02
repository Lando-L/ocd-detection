import tensorflow as tf


class SparseNegativeLogLikelihood(tf.keras.losses.Loss):
    def __init__(self, num_classes: int):
        super(SparseNegativeLogLikelihood, self).__init__()
        self.num_classes = num_classes

    def call(self, y_true, y_pred):
        return -y_pred.log_prob(tf.one_hot(y_true, depth=self.num_classes, dtype=tf.float32))
