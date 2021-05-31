from collections import namedtuple
from functools import partial, reduce
import os
from typing import Callable, Dict, Iterable, List, Text, Tuple

import matplotlib.pylab as plt
import mlflow
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
import tensorflow_federated as tff

from ocddetection import metrics
from ocddetection.data import preprocessing
from ocddetection.types import Metrics, ServerState, FederatedDataset
from ocddetection.learning.federated import common


Config = namedtuple(
  'Config',
  ['path', 'batch_size', 'window_size', 'pos_weight', 'hidden_size']
)

def __load_data(path: Text, window_size: int, batch_size: int) -> FederatedDataset:
  files = pd.Series(
    [os.path.join(path, f'S{subject}-ADL4-AUGMENTED.csv') for subject in range(1, 5)],
    index=list(range(1, 5)),
    name='path'
  )

  return preprocessing.to_federated(files, 1, window_size, batch_size)


def __metrics_fn() -> List[tf.keras.metrics.Metric]:
  thresholds = np.linspace(0, 1, 200)

  return [
    metrics.AUC(from_logits=True, curve='PR', name='pr_auc'),
    metrics.Precision(from_logits=True, thresholds=thresholds, name='precision'),
    metrics.Recall(from_logits=True, thresholds=thresholds, name='recall')
  ]

def __evaluate(
	weights: tff.learning.ModelWeights,
	dataset: FederatedDataset,
	evaluation_fn: Callable[[tff.learning.ModelWeights, List[tf.data.Dataset]], Tuple[tf.Tensor, Dict[Text, tf.Tensor]]]
) -> None:
	confusion_matrix, metrics = evaluation_fn(
		weights,
		[dataset.data[client] for client in dataset.clients]
	)
