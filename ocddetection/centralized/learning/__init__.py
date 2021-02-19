from argparse import ArgumentParser
from functools import partial
from typing import Callable, Dict, List, Tuple

import mlflow
import numpy as np
import tensorflow as tf

from ocddetection import models, callbacks
from ocddetection.data import preprocessing
from ocddetection.types import Metrics


def __arg_parser() -> ArgumentParser:
    parser = ArgumentParser()
    
    # Data
    parser.add_argument('path', type=str)

    # Hyperparameter
    parser.add_argument('--learning-rate', type=float, default=.1)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--window-size', type=int, default=30)

    # Model
    parser.add_argument('--hidden-size', type=int, default=64)
    parser.add_argument('--dropout-rate', type=float, default=.6)

    return parser


def __load_data(path, epochs, window_size, batch_size) -> Tuple[tf.data.Dataset, tf.data.Dataset]:
    table = tf.lookup.StaticHashTable(
        tf.lookup.KeyValueTensorInitializer(list(preprocessing.LABEL2IDX.keys()), list(preprocessing.LABEL2IDX.values())),
        0
    )

    df_train, df_val, _ = preprocessing.split(
        preprocessing.files(path),
        validation=[(1, 1), (2, 1), (3, 1), (4, 1)],
        test=[]
    )

    train = preprocessing \
        .to_centralised(df_train) \
        .map(partial(preprocessing.preprocess, sensors=preprocessing.SENSORS, label=preprocessing.MID_LEVEL, table=table)) \
        .filter(preprocessing.filter_nan) \
        .window(window_size, shift=window_size // 2) \
        .flat_map(partial(preprocessing.windows, window_size=window_size)) \
        .batch(batch_size)

    val = preprocessing \
        .to_centralised(df_val) \
        .map(partial(preprocessing.preprocess, sensors=preprocessing.SENSORS, label=preprocessing.MID_LEVEL, table=table)) \
        .filter(preprocessing.filter_nan) \
        .window(window_size, shift=window_size // 2) \
        .flat_map(partial(preprocessing.windows, window_size=window_size)) \
        .batch(batch_size)

    return train, val


def __model_fn(window_size: int, hidden_size: int, dropout_rate: float) -> tf.keras.Model:     
    return models.bidirectional(
        window_size,
        len(preprocessing.SENSORS),
        len(preprocessing.LABEL2IDX),
        hidden_size,
        dropout_rate
    )


def __optimizer_fn(learning_rate: float) -> tf.keras.optimizers.Optimizer:
    return tf.keras.optimizers.SGD(learning_rate, momentum=.9)


def run(experiment_name: str, run_name: str) -> None:
    mlflow.set_experiment(experiment_name)
    args = __arg_parser().parse_args()

    train, val = __load_data(args.path, args.epochs, args.window_size, args.batch_size)
    
    model = __model_fn(args.window_size, args.hidden_size, args.dropout_rate)
    model.compile(
        optimizer=__optimizer_fn(args.learning_rate),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name='accuracy')]
    )

    with mlflow.start_run(run_name=run_name):
        mlflow.log_params(vars(args))

        model.fit(
            train,
            epochs=args.epochs,
            validation_data=val,
            verbose=0,
            callbacks=[callbacks.MlFlowLogging()]
        )
