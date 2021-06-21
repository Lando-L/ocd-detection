from collections import OrderedDict
import csv
from functools import reduce
from itertools import chain
from typing import List, Text, Tuple

import numpy as np
import pandas as pd
import tensorflow as tf

from ocddetection.types import FederatedDataset
from ocddetection.data import SENSORS


def __read_csv(paths: List[Text]) -> tf.data.Dataset:
    def read(path):
        with open(path, 'r', newline='') as file:
            reader = csv.reader(file, delimiter=',', quoting=csv.QUOTE_NONNUMERIC)
            for row in reader:
                yield row
    
    return tf.data.Dataset.from_generator(
        lambda: chain.from_iterable(map(read, paths)),
        output_types=tf.float32,
        output_shapes=len(SENSORS) + 1
    )


def __preprocess(ds: tf.data.Dataset, epochs: int, window_size: int, batch_size: int) -> tf.data.Dataset:
    def split(t: tf.Tensor):
        return t[:-1], t[-1]
    
    def flatten(X: tf.data.Dataset, y: tf.data.Dataset):
        X = X.batch(window_size, drop_remainder=True)
        y = y.batch(window_size, drop_remainder=True)

        return tf.data.Dataset.zip((X, y))
    
    def label(X: tf.Tensor, y: tf.Tensor):
        return X, tf.expand_dims(tf.round(tf.math.reduce_mean(y)), axis=-1)

    return ds \
        .map(split) \
        .window(window_size, shift=window_size // 2) \
        .flat_map(flatten) \
        .map(label) \
        .batch(batch_size) \
        .repeat(epochs)


def split(paths: pd.Series, validation: List[Tuple[int, int]], test: List[Tuple[int, int]]) -> Tuple[pd.Series, pd.Series, pd.Series]:
    ss_val = paths.loc[validation]
    ss_test = paths.loc[test]
    ss_train = paths.drop(index=validation+test)

    return ss_train, ss_val, ss_test


def to_centralised(paths: pd.DataFrame, window_size: int, batch_size: int) -> tf.data.Dataset:
    client_paths = paths.groupby(level=[0]).agg(list).to_dict()
    client_datasets = [__preprocess(__read_csv(paths), 1, window_size, batch_size) for paths in client_paths.values()]

    return reduce(lambda merged, client: merged.concatenate(client), client_datasets).prefetch(tf.data.AUTOTUNE)


def to_federated(paths: pd.DataFrame, epochs: int, window_size: int, batch_size: int) -> FederatedDataset:
    client_ids = paths.index.get_level_values(0).unique()
    client_paths = paths.groupby(level=[0]).agg(list).to_dict()
    client_datasets = {client: __read_csv(paths) for client, paths in client_paths.items()}

    return FederatedDataset(
        client_ids,
        OrderedDict({
            client: __preprocess(dataset, epochs, window_size, batch_size).prefetch(tf.data.AUTOTUNE)
            for client, dataset in client_datasets.items()
        })
    )
