from typing import List, Tuple

import pandas as pd
import tensorflow as tf

from ocddetection.types import FederatedDataset


def split(df: pd.DataFrame, validation: List[Tuple[int, int]], test: List[Tuple[int, int]]) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df_val = df.loc[validation]
    df_test = df.loc[test]
    df_train = df.drop(index=validation+test)

    return df_train, df_val, df_test


def to_centralised(df: pd.DataFrame) -> tf.data.Dataset:
    return to_dataset(df['path'].to_numpy())


def to_federated(df: pd.DataFrame) -> FederatedDataset:
    client_ids = df.index.get_level_values(0).unique()
    client_paths = df.groupby(level=[0]).agg(list)['path'].to_dict()
    client_datasets = {client: to_dataset(paths) for client, paths in client_paths.items()}

    return FederatedDataset(client_ids, client_datasets)


def preprocess(t: tf.Tensor, sensors: List[int], label: int, table: tf.lookup.StaticHashTable):
    X = tf.cast(tf.gather(t, sensors), tf.float32)
    y = table.lookup(tf.cast(t[label], tf.int32))

    return X, y


def filter_nan(X, y):
    return not tf.math.reduce_any(
        tf.math.is_nan(
            tf.concat((X, tf.cast(tf.expand_dims(y, axis=-1), tf.float32)), axis=-1)
        )
    )


def windows(X, y, window_size: int):
    return tf.data.Dataset.zip((X.batch(window_size, drop_remainder=True), y.batch(window_size, drop_remainder=True)))
