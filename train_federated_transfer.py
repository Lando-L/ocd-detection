import argparse
from collections import OrderedDict
from functools import partial
import time

import mlflow
import pandas as pd
import tensorflow as tf
import tensorflow_federated as tff

from ocddetection import models
from ocddetection.data import preprocessing
from ocddetection.federated.personalization.transfer import process


def arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    
    # Data
    parser.add_argument('path', type=str)

    # Evaluation
    parser.add_argument('--evaluation-rate', type=int, default=10)

    # Hyperparameter
    parser.add_argument('--learning-rate', type=float, default=.1)
    parser.add_argument('--rounds', type=int, default=50)
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--window-size', type=int, default=30)

    # Model
    parser.add_argument('--hidden-size', type=int, default=64)
    parser.add_argument('--dropout-rate', type=float, default=.6)

    return parser


def main():
    mlflow.set_experiment('HAR Federated Transfer')
    
    args = arg_parser().parse_args()

    # Data
    print('Reading files from {}'.format(args.path))
    table = tf.lookup.StaticHashTable(
        tf.lookup.KeyValueTensorInitializer(list(preprocessing.LABEL2IDX.keys()), list(preprocessing.LABEL2IDX.values())),
        0
    )

    df_train, df_val, _ = preprocessing.split(
        preprocessing.files(args.path),
        validation=[(1, 2)],
        test=[(2, 3), (2, 4), (3, 3), (3, 4)]
    )

    train_clients, train_dict = preprocessing.to_federated(df_train)
    train = {
        idx: dataset \
            .map(partial(preprocessing.preprocess, sensors=preprocessing.SENSORS, label=preprocessing.MID_LEVEL, table=table)) \
            .filter(preprocessing.filter_nan) \
            .window(args.window_size, shift=args.window_size // 2) \
            .flat_map(partial(preprocessing.windows, window_size=args.window_size)) \
            .batch(args.batch_size)
        
        for idx, dataset in train_dict.items()
    }

    val_clients, val_dict = preprocessing.to_federated(df_val)
    val = {
        idx: dataset \
            .map(partial(preprocessing.preprocess, sensors=preprocessing.SENSORS, label=preprocessing.MID_LEVEL, table=table)) \
            .filter(preprocessing.filter_nan) \
            .window(args.window_size, shift=args.window_size // 2) \
            .flat_map(partial(preprocessing.windows, window_size=args.window_size)) \
            .batch(args.batch_size)
        
        for idx, dataset in val_dict.items()
    }

    # Model
    def model_fn():
        return models.bidirectional(
            args.window_size,
            len(preprocessing.SENSORS),
            len(preprocessing.LABEL2IDX),
            args.hidden_size,
            args.dropout_rate
        )

    def federated_model_fn():
        return tff.learning.from_keras_model(
            model_fn(),
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            input_spec=(
                tf.TensorSpec((None, args.window_size, len(preprocessing.SENSORS)), dtype=tf.float32),
                tf.TensorSpec((None, args.window_size), dtype=tf.int32)
            ),
            metrics=[
                tf.keras.metrics.SparseCategoricalAccuracy(name='accuracy')
            ]
        )
    
    # Optimizer
    def server_optimizer_fn():
        return tf.keras.optimizers.SGD(1.0, momentum=.9)
    
    def client_optimizer_fn():
        return tf.keras.optimizers.SGD(args.learning_rate)

    # Training Loop
    tff.backends.native.set_local_execution_context(
        server_tf_device=tf.config.list_logical_devices('CPU')[0],
        client_tf_devices=tf.config.list_logical_devices('GPU')
    )

    iterator = tff.learning.build_federated_averaging_process(model_fn, client_optimizer_fn, server_optimizer_fn)

    personalize_fn_dict = OrderedDict(
        sgd=partial(
            process.personalizor,
            optimizer_fn=client_optimizer_fn(),
            batch_size=args.batch_size,
            epochs=args.epochs,
            evaluation_rate=args.evaluation_rate
        )
    )
    
    evaluator = tff.learning.build_personalization_eval(
        model_fn=model_fn,
        personalize_fn_dict=personalize_fn_dict,
        baseline_evaluate_fn=process.evaluate
    )

    state = iterator.initialize()

    with mlflow.start_run():
        mlflow.log_params(vars(args))

        print(f'Training for {args.rounds} rounds a {args.epochs} epoch(s)')
        
        for round_num in range(1, args.rounds + 1):
            start = time.time()
            state, _ = iterator.next(state, list(train.values()))
            
            print(f'Round {round_num}')
            print(f'Time taken for 1 round: {time.time() - start} secs\n')
        
        print(f'Evaluating transfer learn for {args.epochs} epoch(s)')
        metrics = evaluator(state.model)


if __name__ == "__main__":
    main()
