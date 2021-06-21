from argparse import ArgumentParser

import tensorflow as tf
import tensorflow_federated as tff

from ocddetection.learning.federated.stateful import validation
from ocddetection.learning.federated.stateful.impl import layers


def __arg_parser() -> ArgumentParser:
    parser = ArgumentParser()
    
    # Data
    parser.add_argument('path', type=str)

    # Hyperparameter
    parser.add_argument('--rounds', type=int, default=50)
    parser.add_argument('--clients-per-round', type=int, default=4)
    parser.add_argument('--learning-rate', type=float, default=.001)
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--window-size', type=int, default=150)
    parser.add_argument('--pos-weight', type=float, default=6.5)

    # Model
    parser.add_argument('--hidden-size', type=int, default=64)

    return parser


def main() -> None:
    args = __arg_parser().parse_args()

    tff.backends.native.set_local_execution_context(
        server_tf_device=tf.config.list_logical_devices('CPU')[0],
        client_tf_devices=tf.config.list_logical_devices('GPU')
    )

    validation.run(
        'Crossvalidation',
        'Federated Personal Layers',
        layers.setup,
        validation.Config(**vars(args))
    )


if __name__ == "__main__":
    main()
