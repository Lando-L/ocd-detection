from collections import namedtuple
from functools import partial, reduce
import os
from typing import List, Tuple

import matplotlib.pylab as plt
import mlflow
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf

from ocddetection import losses, metrics, models
from ocddetection.data import preprocessing, SENSORS


Config = namedtuple(
  'Config',
  ['path', 'output', 'checkpoint_rate', 'learning_rate', 'epochs', 'batch_size', 'window_size', 'pos_weight', 'hidden_size', 'dropout']
)


def __load_data(path, window_size, batch_size) -> Tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]:
  files = pd.Series(
    [os.path.join(path, f'S{subject}-ADL{run}-AUGMENTED.csv') for subject in range(1, 5) for run in range(1, 6)],
    index=pd.MultiIndex.from_product([list(range(1, 5)), list(range(1, 6))]),
    name='path'
  )

  train_files, _, test_files = preprocessing.split(
    files,
    validation=[],
    test=[(subject, 5) for subject in range(1, 5)]
  )

  train = preprocessing.to_centralised(train_files, window_size, batch_size)
  val = preprocessing.to_centralised(test_files, window_size, batch_size)
  test = preprocessing.to_federated(test_files, window_size, batch_size)

  return train, val, test


def __model_fn(window_size: int, hidden_size: int, dropout: float) -> tf.keras.Model:     
  return models.bidirectional(window_size, len(SENSORS), hidden_size, dropout)


def __training_metrics_fn() -> List[tf.keras.metrics.Metric]:
  return [
		metrics.SigmoidDecorator(tf.keras.metrics.AUC(curve='PR'), name='auc')
	]


def __evaluation_metrics_fn() -> List[tf.keras.metrics.Metric]:
	thresholds = list(np.linspace(0, 1, 200, endpoint=False))
	return [
		metrics.SigmoidDecorator(tf.keras.metrics.AUC(curve='PR'), name='auc'),
		metrics.SigmoidDecorator(tf.keras.metrics.Precision(thresholds=thresholds), name='precision'),
		metrics.SigmoidDecorator(tf.keras.metrics.Recall(thresholds=thresholds), name='recall'),
		metrics.SigmoidDecorator(tf.keras.metrics.BinaryAccuracy(), name='accuracy'),
  ]


def __optimizer_fn(learning_rate: float) -> tf.keras.optimizers.Optimizer:
  return tf.keras.optimizers.SGD(learning_rate=learning_rate)


def __train_step(
  X: tf.Tensor,
  y: tf.Tensor,
  model: tf.keras.Model,
  optimizer: tf.keras.optimizers.Optimizer,
  loss_fn: tf.keras.losses.Loss,
  loss: tf.keras.metrics.Mean,
  metrics: List[tf.keras.metrics.Metric]
) -> None:
  with tf.GradientTape() as tape:
    logits = model(X, training=True)
    loss_value = loss_fn(y, logits)

  optimizer.apply_gradients(
    zip(
      tape.gradient(loss_value, model.trainable_variables),
      model.trainable_variables
    )
  )

  loss.update_state(loss_value)
  
  for metric in metrics:
    metric.update_state(y, logits)


def __validation_step(
  X: tf.Tensor,
  y: tf.Tensor,
  model: tf.keras.Model,
  loss_fn: tf.keras.losses.Loss,
  loss: tf.keras.metrics.Mean,
  metrics: List[tf.keras.metrics.Metric]
) -> None:
  logits = model(X, training=False)
  loss_value = loss_fn(y, logits)

  loss.update_state(loss_value)
  
  for metric in metrics:
    metric.update_state(y, logits)


def __evaluation_step(
  state: tf.Tensor,
  batch: Tuple[tf.Tensor, tf.Tensor],
  model: tf.keras.Model,
  metrics: List[tf.keras.metrics.Metric]
) -> tf.Tensor:
  logits = model(batch[0], training=False)
  y_true = tf.reshape(batch[1], (-1,))
  y_pred = tf.round(tf.nn.sigmoid(tf.reshape(logits, (-1,))))
  
  for metric in metrics:
    metric.update_state(batch[1], logits)
  
  return state + tf.math.confusion_matrix(y_true, y_pred, num_classes=2)


def run(experiment_name: str, run_name: str, config: Config) -> None:
  mlflow.set_experiment(experiment_name)
  train, val, test = __load_data(config.path, config.window_size, config.batch_size)
  
  model = __model_fn(config.window_size, config.hidden_size, config.dropout)
  loss_fn = losses.WeightedBinaryCrossEntropy(config.pos_weight)
  optimizer = __optimizer_fn(config.learning_rate)
  
  input_spec = (
    tf.TensorSpec((None, config.window_size, len(SENSORS)), dtype=tf.float32),
    tf.TensorSpec((None, 1), dtype=tf.float32)
  )

  ckpt = tf.train.Checkpoint(model=model, optimizer=optimizer)
  ckpt_manager = tf.train.CheckpointManager(ckpt, config.output, max_to_keep=5)

  train_loss = tf.keras.metrics.Mean(name='loss')
  train_metrics = __training_metrics_fn()
  train_step = tf.function(
    partial(__train_step, model=model, optimizer=optimizer, loss_fn=loss_fn, loss=train_loss, metrics=train_metrics),
    input_signature=input_spec
  )

  val_loss = tf.keras.metrics.Mean(name='loss')
  val_metrics = __training_metrics_fn()
  val_step = tf.function(
    partial(__validation_step, model=model, loss_fn=loss_fn, loss=val_loss, metrics=val_metrics),
    input_signature=input_spec
  )

  eval_state = tf.zeros((2, 2), dtype=tf.int32)
  eval_metrics = __evaluation_metrics_fn()
  eval_step = partial(__evaluation_step, model=model, metrics=eval_metrics)

  with mlflow.start_run(run_name=run_name):
    mlflow.log_params(config._asdict())

    # Fitting
    for epoch in range(1, config.epochs + 1):
      train_loss.reset_states()
      for metrics in train_metrics:
        metrics.reset_states()

      val_loss.reset_states()
      for metric in val_metrics:
        metric.reset_states()

      # Training
      for X, y in train:
        train_step(X, y)

      mlflow.log_metric(train_loss.name, train_loss.result().numpy(), step=epoch)
      mlflow.log_metrics({metric.name: metric.result().numpy() for metric in train_metrics}, step=epoch)
      
      # Validation
      for X, y in val:
        val_step(X, y)

      mlflow.log_metric(f'val_{val_loss.name}', val_loss.result().numpy(), step=epoch)
      mlflow.log_metrics({f'val_{metric.name}': metric.result().numpy() for metric in val_metrics}, step=epoch)

      # Checkpoint
      if epoch % config.checkpoint_rate == 0:
        ckpt_manager.save()
    
    # Evaluation
    def evaluate(confusion_matrix, client):
      # Reset PR-AUC and Accuracy metrics
      eval_metrics[0].reset_states()
      eval_metrics[3].reset_states()

      results = test[client].reduce(eval_state, eval_step)
      mlflow.log_metrics(f'client_{client}_val_auc', eval_metrics[0].result().numpy())
      mlflow.log_metrics(f'client_{client}_val_acc', eval_metrics[3].result().numpy())

      return confusion_matrix + results

    confusion_matrix = reduce(
      evaluate,
      test.clients,
      tf.zeros((2, 2), dtype=tf.int32)
    )

    # Confusion matrix
    fig, ax = plt.subplots(figsize=(16, 8))

    sns.heatmap(confusion_matrix, annot=True, fmt='d', cmap=sns.color_palette("Blues"), ax=ax)
    
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Ground Truth')

    mlflow.log_figure(fig, 'confusion_matrix.png')
    plt.close(fig)

    # Precision Recall
    fig, ax = plt.subplots(figsize=(16, 8))
    sns.lineplot(x=eval_metrics[2].results().numpy(), y=eval_metrics[1].results().numpy(), ax=ax)

    ax.set_xlabel('Recall')
    ax.set_xlim(0., 1.)

    ax.set_ylabel('Precision')
    ax.set_ylim(0., 1.)

    mlflow.log_figure(fig, 'precision_recall.png')
    plt.close(fig)
