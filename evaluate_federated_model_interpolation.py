from argparse import ArgumentParser
from ocddetection.learning.federated.stateful import evaluation
from ocddetection.learning.federated.stateful.impl import interpolation


def __arg_parser() -> ArgumentParser:
  parser = ArgumentParser()
  
  # Data
  parser.add_argument('path', type=str)
  parser.add_argument('output', type=str)

  # Hyperparameter
  parser.add_argument('--batch-size', type=int, default=128)
  parser.add_argument('--window-size', type=int, default=150)

  # Model
  parser.add_argument('--hidden-size', type=int, default=64)

  return parser


def main() -> None:
    args = __arg_parser().parse_args()

    evaluation.run(
        'Subject Validation',
        'Federated Model Interpolation',
        interpolation.create,
        evaluation.Config(**vars(args))
    )


if __name__ == "__main__":
    main()
