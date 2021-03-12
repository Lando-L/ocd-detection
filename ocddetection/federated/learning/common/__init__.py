from argparse import ArgumentParser
import os
from typing import Tuple

import numpy as np
import pandas as pd
import tensorflow as tf

from ocddetection.data import preprocessing
from ocddetection.types import FederatedDataset, Metrics


def arg_parser() -> ArgumentParser:
    parser = ArgumentParser()
    
    # Data
    parser.add_argument('path', type=str)

    # Evaluation
    parser.add_argument('--validation-rate', type=int, default=5)

    # Hyperparameter
    parser.add_argument('--clients-per-round', type=int, default=3)
    parser.add_argument('--learning-rate', type=float, default=.1)
    parser.add_argument('--rounds', type=int, default=50)
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--window-size', type=int, default=30)

    # Model
    parser.add_argument('--hidden-size', type=int, default=64)
    parser.add_argument('--dropout-rate', type=float, default=.6)

    return parser


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
    val = preprocessing.to_federated(val_files, epochs, window_size, batch_size)
    test = preprocessing.to_federated(test_files, epochs, window_size, batch_size)

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
