import numpy as np

from ocddetection.types import FederatedDataset, Metrics


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
