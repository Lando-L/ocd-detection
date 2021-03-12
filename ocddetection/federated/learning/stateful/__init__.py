from functools import partial, reduce
from typing import Callable, Dict, List, Tuple

import matplotlib.pylab as plt
import mlflow
import seaborn as sns
import tensorflow as tf
import tensorflow_federated as tff

from ocddetection.types import Metrics, ServerState, ClientState, FederatedDataset
from ocddetection.federated.learning import common


def __train_step(
    server_state: ServerState,
    client_states: Dict[int, ClientState],
    dataset: FederatedDataset,
    clients_per_round: int,
    client_idx2id: Dict[int, int],
    training_fn: Callable[[ServerState, List[tf.data.Dataset], List[ClientState]], Tuple[ServerState, Metrics, List[ClientState]]]
) -> Tuple[ServerState, Metrics, Dict[int, ClientState]]:
    sampled_clients = common.sample_clients(dataset, clients_per_round)
    sampled_data = [dataset.data[client] for client in sampled_clients]
    sampled_client_states = [client_states[client] for client in sampled_clients]
    
    next_state, metrics, updated_client_states = training_fn(server_state, sampled_data, sampled_client_states)

    next_client_states = {
        **client_states,
        **{
            client_idx2id[client.client_index.numpy()]: client
            for client in updated_client_states
        }
    }

    return next_state, metrics, next_client_states


def __validation_step(
    weights: tff.learning.ModelWeights,
    client_states: ClientState,
    dataset: FederatedDataset,
    validation_fn: Callable[[tff.learning.ModelWeights, List[tf.data.Dataset], List[ClientState]], Metrics]
) -> Metrics:
    return common.update_test_metrics(
        validation_fn(
            weights,
            [dataset.data[client] for client in dataset.clients],
            [client_states[client] for client in dataset.clients]
        )
    )


def __fit(
    validation_round_rate: int,
    train_step_fn: Callable[[ServerState, Dict[int, ClientState]], Tuple[ServerState, Metrics, Dict[int, ClientState]]],
    validation_step_fn: Callable[[tff.learning.ModelWeights, Dict[int, ClientState]], Metrics]
) -> Callable[[Tuple[ServerState, Dict[int, ClientState]], int], Tuple[ServerState, Dict[int, ClientState]]]:
    def __reduce_fn(aggregates: Tuple[ServerState, Dict[int, ClientState]], round_num: int) -> Tuple[ServerState, Dict[int, ClientState]]:
        state, metrics, client_states = train_step_fn(aggregates[0], aggregates[1])
        mlflow.log_metrics(metrics, step=round_num)

        if round_num % validation_round_rate == 0:
            mlflow.log_metrics(
                validation_step_fn(state.model, client_states),
                step=round_num
            )

        return state, client_states
    
    return __reduce_fn


def __evaluate(
    weights: tff.learning.ModelWeights,
    client_states: ClientState,
    dataset: FederatedDataset,
    evaluation_fn: Callable[[tff.learning.ModelWeights, List[tf.data.Dataset], List[ClientState]], tf.Tensor]
) -> None:
    confusion_matrix = evaluation_fn(
        weights,
        [dataset.data[client] for client in dataset.clients],
        [client_states[client] for client in dataset.clients]
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
    setup_fn: Callable[[int, int, float, float, Dict[int, int]], Tuple[Dict[int, ClientState], Callable, Callable, Callable]]
) -> None:
    mlflow.set_experiment(experiment_name)

    args = common.arg_parser().parse_args()
    
    train, val, _ = common.load_data(args.path, args.epochs, args.window_size, args.batch_size)

    client_idx2id = list(train.clients.union(val.clients))
    client_id2idx = {i: idx for idx, i in enumerate(client_idx2id)}
    client_states, iterator, validator, evaluator = setup_fn(
        args.window_size,
        args.hidden_size,
        args.dropout_rate,
        args.learning_rate,
        client_id2idx
    )

    train_step = partial(
        __train_step,
        dataset=train,
        clients_per_round=args.clients_per_round,
        client_idx2id=client_idx2id,
        training_fn=iterator.next
    )

    validation_step = partial(
        __validation_step,
        dataset=val,
        validation_fn=validator
    )

    fitting_fn = __fit(
        args.validation_rate,
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

        state, client_states = reduce(
            fitting_fn,
            range(1, args.rounds + 1),
            (iterator.initialize(), client_states)
        )

        evaluation_fn(state.model, client_states)
