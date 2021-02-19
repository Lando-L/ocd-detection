from argparse import ArgumentParser
from collections import OrderedDict
from functools import partial
from typing import Tuple

import numpy as np
import tensorflow as tf

from ocddetection.data import preprocessing
from ocddetection.types import FederatedDataset, Metrics


def arg_parser() -> ArgumentParser:
    parser = ArgumentParser()
    
    # Data
    parser.add_argument('path', type=str)

    # Evaluation
    parser.add_argument('--evaluation-rate', type=int, default=5)

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


def load_data(path, epochs, window_size, batch_size) -> Tuple[FederatedDataset, FederatedDataset]:
    table = tf.lookup.StaticHashTable(
        tf.lookup.KeyValueTensorInitializer(list(preprocessing.LABEL2IDX.keys()), list(preprocessing.LABEL2IDX.values())),
        0
    )

    df_train, df_val, _ = preprocessing.split(
        preprocessing.files(path),
        validation=[(1, 1), (2, 1), (3, 1), (4, 1)],
        test=[]
    )

    train_clients, train_dict = preprocessing.to_federated(df_train)
    train = FederatedDataset(
        train_clients,
        OrderedDict({
            idx: dataset \
                .map(partial(preprocessing.preprocess, sensors=preprocessing.SENSORS, label=preprocessing.MID_LEVEL, table=table)) \
                .filter(preprocessing.filter_nan) \
                .window(window_size, shift=window_size // 2) \
                .flat_map(partial(preprocessing.windows, window_size=window_size)) \
                .batch(batch_size) \
                .repeat(epochs)
            
            for idx, dataset in train_dict.items()
        })
    )

    val_clients, val_dict = preprocessing.to_federated(df_val)
    val = FederatedDataset(
        val_clients,
        OrderedDict({
            idx: dataset \
                .map(partial(preprocessing.preprocess, sensors=preprocessing.SENSORS, label=preprocessing.MID_LEVEL, table=table)) \
                .filter(preprocessing.filter_nan) \
                .window(window_size, shift=window_size // 2) \
                .flat_map(partial(preprocessing.windows, window_size=window_size)) \
                .batch(batch_size)
            
            for idx, dataset in val_dict.items()
        })
    )

    return train, val


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
