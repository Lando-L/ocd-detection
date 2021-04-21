from argparse import ArgumentParser
from functools import partial
import os
from typing import Callable, List, Tuple

import matplotlib.pylab as plt
from matplotlib.lines import Line2D
import mlflow
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf

from ocddetection import callbacks, losses, metrics, models
from ocddetection.data import preprocessing, SENSORS
from ocddetection.types import Metrics


def __arg_parser() -> ArgumentParser:
  parser = ArgumentParser()
  
  # Data
  parser.add_argument('path', type=str)
  parser.add_argument('output', type=str)

  # Hyperparameter
  parser.add_argument('--learning-rate', type=float, default=.001)
  parser.add_argument('--epochs', type=int, default=50)
  parser.add_argument('--batch-size', type=int, default=128)
  parser.add_argument('--window-size', type=int, default=150)
  parser.add_argument('--pos-weight', type=float, default=2.0)
  parser.add_argument('--checkpoint-rate', type=float, default=5)

  # Model
  parser.add_argument('--hidden-size', type=int, default=64)
  parser.add_argument('--dropout-rate', type=float, default=.6)

  return parser


def __load_data(path, window_size, batch_size) -> Tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]:
  files = pd.Series(
    [os.path.join(path, f'S{subject}-ADL{run}-AUGMENTED.csv') for subject in range(1, 5) for run in range(1, 6)],
    index=pd.MultiIndex.from_product([list(range(1, 5)), list(range(1, 6))]),
    name='path'
  )

  train_files, val_files, test_files = preprocessing.split(
    files,
    validation=[(subject, 4) for subject in range(1, 5)],
    test=[(subject, 5) for subject in range(1, 5)]
  )

  train = preprocessing.to_centralised(train_files, window_size, batch_size)
  val = preprocessing.to_centralised(val_files, window_size, batch_size)
  test = preprocessing.to_centralised(test_files, window_size, batch_size)

  return train, val, test


def __model_fn(window_size: int, hidden_size: int, dropout_rate: float) -> tf.keras.Model:     
  return models.bidirectional(
    window_size,
    len(SENSORS),
    hidden_size,
    dropout_rate
  )


def __training_metrics_fn() -> List[tf.keras.metrics.Metric]:
  return [
    metrics.AUC(from_logits=True, curve='PR', name='pr_auc')
  ]


def __evaluation_metrics_fn() -> List[tf.keras.metrics.Metric]:
  thresholds = list(np.linspace(0, 1, 200))

  return [
    metrics.Precision(from_logits=True, thresholds=thresholds, name='precision'),
    metrics.Recall(from_logits=True, thresholds=thresholds, name='recall')
  ]


def __optimizer_fn(learning_rate: float) -> tf.keras.optimizers.Optimizer:
  return tf.keras.optimizers.SGD(learning_rate=learning_rate, momentum=0.9)


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


def run(experiment_name: str, run_name: str) -> None:
  mlflow.set_experiment(experiment_name)
  args = __arg_parser().parse_args()

  train, val, _ = __load_data(args.path, args.window_size, args.batch_size)
  
  model = __model_fn(args.window_size, args.hidden_size, args.dropout_rate)
  loss_fn = losses.WeightedBinaryCrossEntropy(args.pos_weight)
  optimizer = __optimizer_fn(args.learning_rate)
  
  input_spec = (
    tf.TensorSpec((None, args.window_size, len(SENSORS)), dtype=tf.float32),
    tf.TensorSpec((None, 1), dtype=tf.float32)
  )

  ckpt = tf.train.Checkpoint(model=model, optimizer=optimizer)
  ckpt_manager = tf.train.CheckpointManager(ckpt, args.output, max_to_keep=5)

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
    mlflow.log_params(vars(args))

    # Fitting
    for epoch in range(1, args.epochs + 1):
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
      if epoch % args.checkpoint_rate == 0:
        ckpt_manager.save()
    
    # Evaluation
    confusion_matrix = val.reduce(eval_state, eval_step)

    # Confusion matrix
    fig, ax = plt.subplots(figsize=(8, 8))

    sns.heatmap(confusion_matrix, annot=True, fmt='d', cmap=sns.color_palette("Blues"), ax=ax)
    
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Ground Truth')

    mlflow.log_figure(fig, f'confusion_matrix.png')
    plt.close(fig)

    # Precision Recall
    fig, ax = plt.subplots(figsize=(16, 8))

    thresholds = list(np.linspace(0, 1, 200))
    sns.lineplot(x=thresholds, y=eval_metrics[0].result().numpy(), ax=ax, color='blue')
    sns.lineplot(x=thresholds, y=eval_metrics[1].result().numpy(), ax=ax, color='skyblue')

    ax.legend(
      [Line2D([0], [0], color='blue', lw=4), Line2D([0], [0], color='skyblue', lw=4)],
      ['precision', 'recall']
    )

    ax.set_xlabel('Threshold')

    mlflow.log_figure(fig, f'precision_recall.png')
    plt.close(fig)
