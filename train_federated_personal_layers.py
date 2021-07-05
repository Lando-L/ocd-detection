from argparse import ArgumentParser
from typing import List

import tensorflow as tf
import tensorflow_federated as tff

from ocddetection.learning.federated.simulation import training
from ocddetection.learning.federated.impl import layers


def __arg_parser() -> ArgumentParser:
    parser = ArgumentParser()
    
    # Data
    parser.add_argument('path', type=str)
    parser.add_argument('output', type=str)

    # Hyperparameter
    parser.add_argument('--rounds', type=int, default=50)
    parser.add_argument('--clients-per-round', type=int, default=4)
    parser.add_argument('--checkpoint-rate', type=int, default=5)
    parser.add_argument('--learning-rate', type=float, default=.001)
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--window-size', type=int, default=150)
    parser.add_argument('--pos-weights', type=float, nargs='+', default=[7, 5.25, 3, 0])

    # Model
    parser.add_argument('--hidden-size', type=int, default=64)
    parser.add_argument('--dropout', type=float, default=.2)

    return parser


def main() -> None:
    args = __arg_parser().parse_args()

    if len(args.pos_weights) == 1:
        args.pos_weights = [args.pos_weights[0]] * 4
    
    assert len(args.pos_weights) == 4, 'pos_weights contain a single value or a value for every client'

    tff.backends.native.set_local_execution_context(
        server_tf_device=tf.config.list_logical_devices('CPU')[0],
        client_tf_devices=tf.config.list_logical_devices('GPU')
    )

    training.run(
        'OCD Detection',
        'FedPer',
        layers.setup,
        training.Config(**vars(args))
    )


if __name__ == "__main__":
    main()
