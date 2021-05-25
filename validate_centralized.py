from argparse import ArgumentParser
from ocddetection.learning.centralized import validation


def __arg_parser() -> ArgumentParser:
    parser = ArgumentParser()
    
    # Data
    parser.add_argument('path', type=str)

    # Hyperparameter
    parser.add_argument('--learning-rate', type=float, default=.001)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--window-size', type=int, default=60)
    parser.add_argument('--pos-weight', type=float, default=5.0)

    # Model
    parser.add_argument('--hidden-size', type=int, default=128)

    return parser


def main() -> None:
    args = __arg_parser().parse_args()

    validation.run(
        'Crossvalidation',
        'Centralized',
        validation.Config(**vars(args))
    )


if __name__ == "__main__":
    main()
