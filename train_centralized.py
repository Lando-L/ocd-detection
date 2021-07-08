from argparse import ArgumentParser
from ocddetection.learning.centralized import training


def __arg_parser() -> ArgumentParser:
  parser = ArgumentParser()
  
  # Data
  parser.add_argument('path', type=str)
  parser.add_argument('output', type=str)

  # Hyperparameter
  parser.add_argument('--checkpoint-rate', type=float, default=5)
  parser.add_argument('--learning-rate', type=float, default=.001)
  parser.add_argument('--epochs', type=int, default=50)
  parser.add_argument('--batch-size', type=int, default=128)
  parser.add_argument('--window-size', type=int, default=150)
  parser.add_argument('--pos-weight', type=float, default=7.3)

  # Model
  parser.add_argument('--hidden-size', type=int, default=64)
  parser.add_argument('--dropout', type=float, default=.2)

  return parser


def main() -> None:
    args = __arg_parser().parse_args()

    training.run(
        'OCD Detection',
        'Centralized',
        training.Config(**vars(args))
    )


if __name__ == "__main__":
    main()
