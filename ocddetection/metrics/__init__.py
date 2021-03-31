import tensorflow as tf

class AUC(tf.keras.metrics.AUC):
  def __init__(self, from_logits=False, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self._from_logits = from_logits

  def update_state(self, y_true, y_pred, sample_weight=None):
    if self._from_logits:
      super(AUC, self).update_state(y_true, tf.nn.sigmoid(y_pred), sample_weight)
    else:
      super(AUC, self).update_state(y_true, y_pred, sample_weight)

class Precision(tf.keras.metrics.Precision):
  def __init__(self, from_logits=False, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self._from_logits = from_logits

  def update_state(self, y_true, y_pred, sample_weight=None):
    if self._from_logits:
      super(Precision, self).update_state(y_true, tf.nn.sigmoid(y_pred), sample_weight)
    else:
      super(Precision, self).update_state(y_true, y_pred, sample_weight)

class Recall(tf.keras.metrics.Recall):
  def __init__(self, from_logits=False, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self._from_logits = from_logits

  def update_state(self, y_true, y_pred, sample_weight=None):
    if self._from_logits:
      super(Recall, self).update_state(y_true, tf.nn.sigmoid(y_pred), sample_weight)
    else:
      super(Recall, self).update_state(y_true, y_pred, sample_weight)
