import argparse
from functools import partial

import pandas as pd
import tensorflow as tf
import tensorflow_federated as tff

from ocddetection import data, federated


def arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    
    # Data
    parser.add_argument('path', type=str)

    # Hyperparameter
    parser.add_argument('--learning-rate', type=float, default=.1)
    parser.add_argument('--rounds', type=int, default=25)
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--window-size', type=int, default=30)
    parser.add_argument('--keep-state-rate', type=float, default=.5)

    # Model
    parser.add_argument('--hidden-size', type=int, default=64)
    parser.add_argument('--dropout-rate', type=float, default=.4)

    return parser


def main():
    args = arg_parser().parse_args()

    # Data
    print('Reading files from {}'.format(args.path))
    table = tf.lookup.StaticHashTable(
        tf.lookup.KeyValueTensorInitializer(list(data.LABEL2IDX.keys()), list(data.LABEL2IDX.values())),
        0
    )

    df_train, df_val, _ = data.split(
        data.files(args.path),
        validation=[(1, 2)],
        test=[(2, 3), (2, 4), (3, 3), (3, 4)]
    )

    train_clients, train_dict = data.to_federated(df_train)
    train = {
        idx: dataset \
            .map(partial(data.preprocess, sensors=data.SENSORS, label=data.MID_LEVEL, table=table)) \
            .filter(data.filter_nan) \
            .window(args.window_size, shift=args.window_size // 2) \
            .flat_map(partial(data.windows, window_size=args.window_size)) \
            .batch(args.batch_size, drop_remainder=True)
            .repeat(args.epochs)
        
        for idx, dataset in train_dict.items()
    }

    val_clients, val_dict = data.to_federated(df_val)
    val = {
        idx: dataset \
            .map(partial(data.preprocess, sensors=data.SENSORS, label=data.MID_LEVEL, table=table)) \
            .filter(data.filter_nan) \
            .window(args.window_size, shift=args.window_size // 2) \
            .flat_map(partial(data.windows, window_size=args.window_size)) \
            .batch(args.batch_size, drop_remainder=True)
        
        for idx, dataset in val_dict.items()
    }

    # Model
    def model_fn():
        return tff.learning.from_keras_model(
            federated.models.bidirectional(
                args.window_size,
                len(data.SENSORS),
                len(data.LABEL2IDX),
                args.hidden_size,
                args.dropout_rate
            ),
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            input_spec=(
                tf.TensorSpec((None, args.window_size, len(data.SENSORS)), dtype=tf.float32, name='x'),
                tf.TensorSpec((None, args.window_size), dtype=tf.int32, name='y')
            ),
            metrics=[tf.keras.metrics.SparseCategoricalAccuracy()]
        )
    
    # Optimizer
    def server_optimizer_fn():
        return tf.keras.optimizers.SGD(1.0)
    
    def client_optimizer_fn():
        return tf.keras.optimizers.Adam(args.learning_rate)

    # Logs
    train_log_dir = './logs/train'
    train_summary_writer = tf.summary.create_file_writer(train_log_dir)

    # Federtaed Training Loop
    print('Training for {} rounds a {} epochs'.format(args.rounds, args.epochs))

    iterator = federated.iterator(model_fn, server_optimizer_fn, client_optimizer_fn)
    state = iterator.initialize()

    for round in args.rounds:
        state, metrics = iterator.next(state, list(train.values()))

        with train_summary_writer.as_default():
            for name, value in metrics.values():
                tf.summary.scalar(name, value, step=round)


if __name__ == "__main__":
    main()
