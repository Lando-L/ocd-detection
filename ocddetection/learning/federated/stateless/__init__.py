from functools import partial, reduce
from typing import Callable, Dict, List, Text, Tuple

import matplotlib.pylab as plt
from matplotlib.lines import Line2D
import mlflow
import numpy as np
import seaborn as sns
import tensorflow as tf
import tensorflow_federated as tff

from ocddetection.types import Metrics, ServerState, FederatedDataset
from ocddetection.learning.federated import common


def __train_step(
    state: ServerState,
    dataset: FederatedDataset,
    clients_per_round: int,
    training_fn: Callable[[ServerState, List[tf.data.Dataset]], Tuple[ServerState, Metrics]]
) -> Tuple[ServerState, Metrics]:
    sampled_clients = common.sample_clients(dataset, clients_per_round)
    sampled_data = [dataset.data[client] for client in sampled_clients]

    next_state, metrics = training_fn(state, sampled_data)

    return next_state, metrics


def __validation_step(
    weights: tff.learning.ModelWeights,
    dataset: FederatedDataset,
    validation_fn: Callable[[tff.learning.ModelWeights, List[tf.data.Dataset]], Metrics]
) -> Metrics:
    return common.update_test_metrics(
        validation_fn(
            weights,
            [dataset.data[client] for client in dataset.clients]
        )
    )


def __fit(
    state: ServerState,
    round_num: int,
    validation_round_rate: int,
    checkpoint_manager: tff.simulation.FileCheckpointManager,
    train_step_fn: Callable[[ServerState], Tuple[ServerState, Metrics]],
    validation_step_fn: Callable[[tff.learning.ModelWeights], Metrics]
) -> ServerState:
    next_state, metrics = train_step_fn(state)
    mlflow.log_metrics(metrics, step=round_num)

    if round_num % validation_round_rate == 0:
        test_metrics = validation_step_fn(next_state.model)
        mlflow.log_metrics(test_metrics, step=round_num)
        checkpoint_manager.save_checkpoint(next_state, round_num)

    return next_state


def __evaluate(
    weights: tff.learning.ModelWeights,
    dataset: FederatedDataset,
    evaluation_fn: Callable[[tff.learning.ModelWeights, List[tf.data.Dataset]], Tuple[tf.Tensor, Dict[Text, tf.Tensor]]]
) -> None:
    confusion_matrix, metrics = evaluation_fn(
        weights,
        [dataset.data[client] for client in dataset.clients]
    )

    # Confusion Matrix
    fig, ax = plt.subplots(figsize=(16, 8))

    sns.heatmap(confusion_matrix, annot=True, fmt='d', cmap=sns.color_palette("Blues"), ax=ax)
    
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Ground Truth')

    mlflow.log_figure(fig, f'confusion_matrix.png')
    plt.close(fig)

    # Precision Recall
    fig, ax = plt.subplots(figsize=(16, 8))

    thresholds = list(np.linspace(0, 1, 200))
    sns.lineplot(x=thresholds, y=metrics['precision'], ax=ax, color='blue')
    sns.lineplot(x=thresholds, y=metrics['recall'], ax=ax, color='skyblue')

    ax.legend(
        [Line2D([0], [0], color='blue', lw=4), Line2D([0], [0], color='skyblue', lw=4)],
        ['precision', 'recall']
    )

    ax.set_xlabel('Threshold')

    mlflow.log_figure(fig, f'precision_recall.png')
    plt.close(fig)


def run(
    experiment_name: str,
    run_name: str,
    setup_fn: Callable[[int, int, int, float, float, float], Tuple[Callable, Callable, Callable]]
) -> None:
    mlflow.set_experiment(experiment_name)
    args = common.arg_parser().parse_args()
    
    train, val, _ = common.load_data(args.path, args.epochs, args.window_size, args.batch_size)

    checkpoint_manager = tff.simulation.FileCheckpointManager(args.output)

    iterator, validator, evaluator = setup_fn(
        args.window_size,
        args.batch_size,
        args.hidden_size,
        args.dropout_rate,
        args.learning_rate,
        args.pos_weight
    )

    train_step = partial(
        __train_step,
        dataset=train,
        clients_per_round=args.clients_per_round,
        training_fn=iterator.next
    )

    validation_step = partial(
        __validation_step,
        dataset=val,
        validation_fn=validator
    )

    fitting_fn = partial(
        __fit,
        validation_round_rate=args.validation_rate,
        checkpoint_manager=checkpoint_manager,
        train_step_fn=train_step,
        validation_step_fn=validation_step
    )

    evaluation_fn = partial(
        __evaluate,
        dataset=val,
        evaluation_fn=evaluator
    )

    with mlflow.start_run(run_name=run_name):
        mlflow.log_params(vars(args))

        state = reduce(
            fitting_fn,
            range(1, args.rounds + 1),
            iterator.initialize()
        )

        evaluation_fn(state.model)
