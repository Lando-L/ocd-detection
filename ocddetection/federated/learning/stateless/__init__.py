from functools import partial, reduce
from typing import Callable, List, Tuple

import mlflow
import tensorflow as tf
import tensorflow_federated as tff

from ocddetection.types import Metrics, ServerState, FederatedDataset
from ocddetection.federated.learning import common


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
    evaluation_round_rate: int,
    train_step_fn: Callable[[ServerState], Tuple[ServerState, Metrics]],
    validation_step_fn: Callable[[tff.learning.ModelWeights], Metrics]
) -> Callable[[ServerState, int], ServerState]:
    def __reduce_fn(state: ServerState, round_num: int) -> ServerState:
        next_state, metrics = train_step_fn(state)
        mlflow.log_metrics(metrics, step=round_num)

        if round_num % evaluation_round_rate == 0:
            test_metrics = validation_step_fn(next_state.model)
            mlflow.log_metrics(test_metrics, step=round_num)

        return next_state
    
    return __reduce_fn


def __evaluate(
    weights: tff.learning.ModelWeights,
    dataset: FederatedDataset,
    evaluation_fn: Callable[[tff.learning.ModelWeights, List[tf.data.Dataset], List[ClientState]], tf.Tensor]
) -> None:
    confusion_matrix = evaluation_fn(
        weights,
        [dataset.data[client] for client in dataset.clients]
    )

    fig, ax = plt.subplots()

    sns.heatmap(confusion_matrix, annot=True, fmt='d', cmap=sns.color_palette("Blues"), ax=ax)
    
    ax.set_xlabel('Predicted')
    ax.set_ylable('Ground Truth')

    mlflow.log_figure(fig, f'confusion_matrix.png')
    plt.close(fig)


def run(
    experiment_name: str,
    run_name: str,
    setup_fn: Callable[[int, int, float, float], Tuple[Callable, Callable, Callable]]
) -> None:
    mlflow.set_experiment(experiment_name)
    args = common.arg_parser().parse_args()
    
    train, val, _ = common.load_data(args.path, args.epochs, args.window_size, args.batch_size)

    iterator, validator = setup_fn(
        args.window_size,
        args.hidden_size,
        args.dropout_rate,
        args.learning_rate
    )

    train_step = partial(
        __train_step,
        dataset=train,
        clients_per_round=args.clients_per_round,
        train_fn=iterator.next
    )

    validation_step = partial(
        __validation_step,
        dataset=val,
        validation_fn=validator
    )

    fitting_fn = __fit(
        args.evaluation_rate,
        train_step,
        validation_step
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
