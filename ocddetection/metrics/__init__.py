import abc
from typing import Callable

import tensorflow as tf


class MetricsDecorator(tf.keras.metrics.Metric, metaclass=abc.ABCMeta):
  def __init__(self, metric: tf.keras.metrics.Metric, **kwargs):
    super(MetricsDecorator, self).__init__(**kwargs)
    self._metric = metric

  def update_state(self, *args, **kwargs):
    self._metric.update_state(*args, **kwargs)
  
  def get_config(self):
    return {**self._metric.get_config(), '_cls': self._metric.__class__}
  
  @classmethod
  def from_config(cls, config):
    metric_cls = config.pop('_cls', None)

    if cls is MetricsDecorator:
      return cls(metric_cls.from_config(config))
    
    else:
      return super(MetricsDecorator, cls).from_config(config)

  def result(self):
    return self._metric.result()
  
  def reset_states(self):
    self._metric.reset_states()


class LogitsDecorator(MetricsDecorator):
  def __init__(self, metric: tf.keras.metrics.Metric, activation_fn: Callable, **kwargs):
    super(LogitsDecorator, self).__init__(metric, **kwargs)
    self._activation_fn = activation_fn

  def update_state(self, y_true, y_pred, sample_weight=None):
    self._metric.update_state(y_true, self._activation_fn(y_pred), sample_weight)
  
  def get_config(self):
    return {**self._metric.get_config(), '_cls': self._metric.__class__, '_activation_fn': self._activation_fn}
  
  @classmethod
  def from_config(cls, config):
    metric_cls = config.pop('_cls', None)
    metric_fn = config.pop('_activation_fn', None)
    
    if cls is LogitsDecorator:
      return cls(metric_cls.from_config(config), metric_fn)

    else:
      return super(LogitsDecorator, cls).from_config(config)


class SigmoidDecorator(LogitsDecorator):
  def __init__(self, metric: tf.keras.metrics.Metric, **kwargs):
    super(SigmoidDecorator, self).__init__(metric, tf.nn.sigmoid, **kwargs)
      
  def get_config(self):
    return {**self._metric.get_config(), '_cls': self._metric.__class__}
  
  @classmethod
  def from_config(cls, config):
    metric_cls = config.pop('_cls', None)
    
    if cls is SigmoidDecorator:
      return cls(metric_cls.from_config(config))

    else:
      return super(SigmoidDecorator, cls).from_config(config)
