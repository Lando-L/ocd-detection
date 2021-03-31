# from argparse import ArgumentParser
# from typing import Callable, List, Text, Tuple

# import mlflow
# import numpy as np
# import tensorflow as tf

# from ocddetection import metrics


# def __optimizer_fn(learning_rate: float) -> tf.keras.optimizers.Optimizer:
#   return tf.keras.optimizers.Adam(learning_rate=learning_rate)


# def __fitting_metrics_fn() -> List[tf.keras.metrics.Metric]:
#   return [
#     metrics.AUC(from_logits=True, curve='PR', name='pr_auc')
#   ]


# def __evaluation_metrics() -> List[tf.keras.metrics.Metric]:
#   thresholds = list(np.linspace(0, 1, 200))

#   return [
#     metrics.Precision(from_logits=True, thresholds=thresholds, name='precision'),
#     metrics.Recall(from_logits=True, thresholds=thresholds, name='recall')
#   ]


# def run(
#   experiment_name: Text,
#   run_name: Text,
#   arg_parser: ArgumentParser,
#   load_fn: Callable[[Tuple], Tuple],
#   model_fn: Callable[[int, int, float], tf.keras.Model],
#   fitting_fn: Callable[[Tuple], None]
# ) -> None:
#   mlflow.set_experiment(experiment_name)

#   args = arg_parser.parse_args()

#   train, val, _ = load_fn(args)

#   with mlflow.start_run(run_name=run_name):
#     mlflow.log_params(vars(args))

#     # Fitting
#     fitting_fn(model_fntrain, val)
