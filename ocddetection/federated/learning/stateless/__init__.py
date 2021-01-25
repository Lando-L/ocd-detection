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
    train_fn: Callable[[ServerState, List[tf.data.Dataset]], Tuple[ServerState, Metrics]]
) -> Tuple[ServerState, Metrics]:
    sampled_clients = common.sample_clients(dataset, clients_per_round)
    sampled_data = [dataset.data[client] for client in sampled_clients]

    return train_fn(state, sampled_data)


def __evaluation_step(
    weights: tff.learning.ModelWeights,
    dataset: FederatedDataset,
    evaluate_fn: Callable[[tff.learning.ModelWeights, List[tf.data.Dataset]], Metrics]
) -> Metrics:
    return common.update_test_metrics(
        evaluate_fn(
            weights,
            [dataset.data[client] for client in dataset.clients]
        )
    )


def __step(
    evaluation_round_rate: int,
    train_step_fn: Callable[[ServerState], Tuple[ServerState, Metrics]],
    evaluation_step_fn: Callable[[tff.learning.ModelWeights], Metrics]
) -> Callable[[ServerState, int], ServerState]:
    def __reduce_fn(state: ServerState, round_num: int) -> ServerState:
        next_state, metrics = train_step_fn(state)
        mlflow.log_metrics(metrics['train'], step=round_num)

        if round_num % evaluation_round_rate == 0:
            test_metrics = evaluation_step_fn(next_state.model)
            mlflow.log_metrics(test_metrics, step=round_num)

        return next_state
    
    return __reduce_fn


def run(
    experiment_name: str,
    run_name: str,
    setup_fn: Callable[[int, int, float, float], Tuple[Callable, Callable]]
) -> None:
    mlflow.set_experiment(experiment_name)
    args = common.arg_parser().parse_args()
    
    train, val = common.load_data(args.path, args.epochs, args.window_size, args.batch_size)

    iterator, evaluator = setup_fn(
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

    evaluation_step = partial(
        __evaluation_step,
        dataset=val,
        evaluate_fn=evaluator
    )

    reduce_fn = __step(
        args.evaluation_rate,
        train_step,
        evaluation_step
    )

    with mlflow.start_run(run_name=run_name):
        mlflow.log_params(vars(args))

        state = reduce(
            reduce_fn,
            range(1, args.rounds + 1),
            iterator.initialize()
        )
