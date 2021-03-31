import tensorflow as tf

class WeightedBinaryCrossEntropy(tf.keras.losses.Loss):
  def __init__(self, pos_weight, name='weighted_binary_crossentropy'):
    super(WeightedBinaryCrossEntropy, self).__init__(name=name)
    self.pos_weight = pos_weight

  def call(self, y_true, y_pred):
    return tf.nn.weighted_cross_entropy_with_logits(
      y_true,
      y_pred,
      self.pos_weight
    )
