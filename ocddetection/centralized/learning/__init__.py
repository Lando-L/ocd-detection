from argparse import ArgumentParser
import os
from typing import Tuple

import matplotlib.pylab as plt
import mlflow
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf

from ocddetection import models, callbacks, data
from ocddetection.data import preprocessing
from ocddetection.types import Metrics


def __arg_parser() -> ArgumentParser:
    parser = ArgumentParser()
    
    # Data
    parser.add_argument('path', type=str)

    # Hyperparameter
    parser.add_argument('--learning-rate', type=float, default=.01)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--window-size', type=int, default=30)

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
        len(data.SENSORS),
        hidden_size,
        dropout_rate
    )


def __optimizer_fn(learning_rate: float) -> tf.keras.optimizers.Optimizer:
    return tf.keras.optimizers.SGD(learning_rate, momentum=.9)


def run(experiment_name: str, run_name: str) -> None:
    mlflow.set_experiment(experiment_name)
    args = __arg_parser().parse_args()

    train, val, _ = __load_data(args.path, args.window_size, args.batch_size)
    
    model = __model_fn(args.window_size, args.hidden_size, args.dropout_rate)
    model.compile(
        optimizer=__optimizer_fn(args.learning_rate),
        loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
        metrics=[tf.keras.metrics.BinaryAccuracy(name='accuracy')]
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

        y_true = val.map(lambda X, y: y).unbatch()
        
        logits = model.predict(val).reshape((-1, args.window_size))
        y_pred = tf.data.Dataset.from_tensor_slices(tf.round(tf.nn.sigmoid(logits)))

        cm = tf.data.Dataset.zip((y_true, y_pred)).reduce(
            tf.zeros((2, 2), dtype=tf.int32),
            lambda state, t: state + tf.math.confusion_matrix(t[0], t[1], num_classes=2)
        )

        fig, ax = plt.subplots()

        sns.heatmap(cm, annot=True, fmt='d', cmap=sns.color_palette("Blues"), ax=ax)
        
        ax.set_xlabel('Predicted')
        ax.set_ylable('Ground Truth')

        mlflow.log_figure(fig, f'confusion_matrix.png')
        plt.close(fig)
