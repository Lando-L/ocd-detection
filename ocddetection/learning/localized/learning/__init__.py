from argparse import ArgumentParser
from collections import OrderedDict, defaultdict
from functools import partial
from typing import Callable, Dict, List, Tuple

import mlflow
import numpy as np
import tensorflow as tf

from ocddetection import models
from ocddetection.data import preprocessing
from ocddetection.types import FederatedDataset, Metrics
from ocddetection.centralized.learning import callbacks


def __arg_parser() -> ArgumentParser:
    parser = ArgumentParser()
    
    # Data
    parser.add_argument('path', type=str)

    # Hyperparameter
    parser.add_argument('--learning-rate', type=float, default=.1)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--window-size', type=int, default=30)

    # Model
    parser.add_argument('--hidden-size', type=int, default=64)
    parser.add_argument('--dropout-rate', type=float, default=.6)

    return parser


def __load_data(path, epochs, window_size, batch_size) -> Tuple[FederatedDataset, FederatedDataset]:
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
                .batch(batch_size)
            
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


def __train_step_fn(X, y, model, optimizer, loss_fn, loss, metrics) -> None:
    with tf.GradientTape() as tape:
        y_hat = model(X, training=True)
        _loss = loss_fn(y, y_hat)

    optimizer.apply_gradients(
        zip(
            tape.gradient(_loss, model.trainable_variables),
            model.trainable_variables
        )
    )

    loss(_loss)
    for metric in metrics:
        metric(y, y_hat)


def __evaluation_step(X, y, model, loss_fn, loss, metrics) -> None:
    y_hat = model(X, training=False)
    _loss = loss_fn(y, y_hat)

    loss(_loss)
    for metric in metrics:
        metric(y, y_hat)


def __log(key: str, epoch: int, weight: int, value: float, state: Dict[str, Dict[int, List[Tuple[int, float]]]]) -> None:
    state[key][epoch].append((weight, value))


def run(experiment_name: str, run_name: str) -> None:
    mlflow.set_experiment(experiment_name)
    args = __arg_parser().parse_args()

    train, val = __load_data(args.path, args.epochs, args.window_size, args.batch_size)
    
    optimizer = __optimizer_fn(args.learning_rate)
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    with mlflow.start_run(run_name=run_name):
        mlflow.log_params(vars(args))

        logger = defaultdict(lambda: defaultdict(list))
        log = partial(__log, state=logger)
        
        for client in train.clients:
            train_loss = tf.keras.metrics.Mean(name='loss')
            train_metrics = [tf.keras.metrics.SparseCategoricalAccuracy(name='accuracy')]

            val_loss = tf.keras.metrics.Mean(name='val_loss')
            val_metrics = [tf.keras.metrics.SparseCategoricalAccuracy(name='val_accuracy')]

            num_examples = tf.Variable(0, trainable=False, dtype=tf.int32)

            model = __model_fn(args.window_size, args.hidden_size, args.dropout_rate)

            train_step = tf.function(
                partial(
                    __train_step_fn,
                    model=model,
                    optimizer=optimizer,
                    loss_fn=loss_fn,
                    loss=train_loss,
                    metrics=train_metrics
                )
            )

            eval_step = tf.function(
                partial(
                    __evaluation_step,
                    model=model,
                    loss_fn=loss_fn,
                    loss=val_loss,
                    metrics=val_metrics
                )
            )

            for epoch in range(1, args.epochs + 1):
                # Training
                train_loss.reset_states()
                for metric in train_metrics:
                    metric.reset_states()

                num_examples.assign(0)

                for X, y in train.data[client]:
                    train_step(X, y)
                    num_examples.assign_add(tf.shape(y)[0])
                
                log(train_loss.name, epoch, num_examples.read_value().numpy(), train_loss.result().numpy())
                for metric in train_metrics:
                    log(metric.name, epoch, num_examples.read_value().numpy(), metric.result().numpy())
                
                # Validation
                val_loss.reset_states()
                for metric in val_metrics:
                    metric.reset_states()
                
                for X, y in val.data[client]:
                    eval_step(X, y)
            
                log(val_loss.name, epoch, num_examples.read_value().numpy(), val_loss.result().numpy())
                for metric in val_metrics:
                    log(metric.name, epoch, num_examples.read_value().numpy(), metric.result().numpy())
        
        for name, epochs in logger.items():
            for epoch, values in epochs.items():
                weights, results = list(zip(*values))
                mlflow.log_metric(name, np.average(results, weights=weights), step=epoch)
