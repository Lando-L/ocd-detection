from collections import namedtuple
import os
from typing import Tuple

import numpy as np
import pandas as pd

from ocddetection.data import preprocessing
from ocddetection.types import FederatedDataset, Metrics


Config = namedtuple(
  'Config',
  ['path', 'output', 'validation_rate', 'clients_per_round', 'learning_rate', 'rounds', 'epochs', 'batch_size', 'window_size', 'pos_weight', 'hidden_size']
)


def load_data(path, epochs, window_size, batch_size) -> Tuple[FederatedDataset, FederatedDataset, FederatedDataset]:
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


def sample_clients(dataset: FederatedDataset, clients_per_round: int) -> np.ndarray:
	return np.random.choice(
		dataset.clients,
		size=clients_per_round,
		replace=False
	)


def update_test_metrics(metrics: Metrics) -> Metrics:
	return {
		f'val_{name}': metric
		for name, metric in metrics.items()
	}
