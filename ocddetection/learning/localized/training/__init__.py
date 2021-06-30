from collections import namedtuple
from functools import partial
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
from ocddetection.types import FederatedDataset


Config = namedtuple(
  'Config',
  ['path', 'output', 'checkpoint_rate', 'learning_rate', 'epochs', 'batch_size', 'window_size', 'pos_weights', 'hidden_size', 'dropout']
)


def __load_data(path, epochs, window_size, batch_size) -> Tuple[FederatedDataset, FederatedDataset, FederatedDataset]:
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

	train = preprocessing.to_federated(train_files, epochs, window_size, batch_size)
	val = preprocessing.to_federated(val_files, 1, window_size, batch_size)
	test = preprocessing.to_federated(test_files, 1, window_size, batch_size)

	return train, val, test


def __model_fn(window_size: int, hidden_size: int, dropout: float) -> tf.keras.Model:     
  return models.bidirectional(window_size, len(SENSORS), hidden_size, dropout)


def __metrics_fn() -> List[tf.keras.metrics.Metric]:
  thresholds = list(np.linspace(0, 1, 200, endpoint=False))

  return [
    metrics.AUC(from_logits=True, curve='PR', name='auc'),
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
  loss_fn: tf.keras.losses.Loss
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
  train, val, _ = __load_data(config.path, config.window_size, config.batch_size)

  input_spec = (
    tf.TensorSpec((None, config.window_size, len(SENSORS)), dtype=tf.float32),
    tf.TensorSpec((None, 1), dtype=tf.float32)
  )

  with mlflow.start_run(run_name=run_name):
    mlflow.log_params(config._asdict())

    for i, client in enumerate(train.clients):
      model = __model_fn(config.window_size, config.hidden_size, config.dropout)
      loss_fn = losses.WeightedBinaryCrossEntropy(config.pos_weights[i])
      optimizer = __optimizer_fn(config.learning_rate)

      ckpt = tf.train.Checkpoint(model=model, optimizer=optimizer)
      ckpt_manager = tf.train.CheckpointManager(ckpt, os.path.join(config.output, i), max_to_keep=5)

      train_loss = tf.keras.metrics.Mean(name='loss')
      train_step = tf.function(
        partial(__train_step, model=model, optimizer=optimizer, loss_fn=loss_fn, loss=train_loss),
        input_signature=input_spec
      )

      eval_state = tf.zeros((2, 2), dtype=tf.int32)
      eval_metrics = __metrics_fn()
      eval_step = partial(__evaluation_step, model=model, metrics=eval_metrics)

      # Fitting
      for epoch in range(1, config.epochs + 1):
        # Training
        for X, y in train.data[client]:
          train_step(X, y)

        # Checkpoint
        if epoch % config.checkpoint_rate == 0:
          ckpt_manager.save()
      
        # Evaluation
        confusion_matrix = val.data[client].reduce(eval_state, eval_step)

        # Confusion matrix
        fig, ax = plt.subplots(figsize=(16, 8))

        sns.heatmap(confusion_matrix, annot=True, fmt='d', cmap=sns.color_palette("Blues"), ax=ax)
        
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Ground Truth')

        mlflow.log_figure(fig, f'confusion_matrix_{client}.png')
        plt.close(fig)

        # AUC
        mlflow.log_metric(f'val_auc_{client}', eval_metrics[0].result().numpy())

        # Precision Recall
        fig, ax = plt.subplots(figsize=(16, 8))
        sns.lineplot(x=eval_metrics[1].result().numpy(), y=eval_metrics[2].result().numpy(), ax=ax)

        ax.set_xlabel('Recall')
        ax.set_xlim(0., 1.)

        ax.set_ylabel('Precision')
        ax.set_ylim(0., 1.)

        mlflow.log_figure(fig, f'precision_recall_{client}.png')
        plt.close(fig)
