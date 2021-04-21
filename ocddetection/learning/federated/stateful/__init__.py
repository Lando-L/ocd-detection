from functools import partial, reduce
from typing import Callable, Dict, List, Text, Tuple

import matplotlib.pylab as plt
from matplotlib.lines import Line2D
import mlflow
import numpy as np
import seaborn as sns
import tensorflow as tf
import tensorflow_federated as tff

from ocddetection.types import Metrics, ServerState, ClientState, FederatedDataset
from ocddetection.learning.federated import common


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
    state: Tuple[ServerState, Dict[int, ClientState]],
    round_num: int,
    validation_round_rate: int,
    checkpoint_manager: tff.simulation.FileCheckpointManager,
    train_step_fn: Callable[[ServerState, Dict[int, ClientState]], Tuple[ServerState, Metrics, Dict[int, ClientState]]],
    validation_step_fn: Callable[[tff.learning.ModelWeights, Dict[int, ClientState]], Metrics]
) -> ServerState:
    next_state, metrics, next_client_states = train_step_fn(state[0], state[1])
    mlflow.log_metrics(metrics, step=round_num)

    if round_num % validation_round_rate == 0:
        test_metrics = validation_step_fn(next_state.model, next_client_states)
        mlflow.log_metrics(test_metrics, step=round_num)
        checkpoint_manager.save_checkpoint([next_state, next_client_states], round_num)

    return next_state, next_client_states


def __evaluate(
    weights: tff.learning.ModelWeights,
    client_states: ClientState,
    dataset: FederatedDataset,
    evaluation_fn: Callable[[tff.learning.ModelWeights, List[tf.data.Dataset], List[ClientState]], Tuple[tf.Tensor, Dict[Text, tf.Tensor]]]
) -> None:
    confusion_matrix, metrics = evaluation_fn(
        weights,
        [dataset.data[client] for client in dataset.clients],
        [client_states[client] for client in dataset.clients]
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
    setup_fn: Callable[[int, int, int, float, float, float, Dict[int, int]], Tuple[Dict[int, ClientState], Callable, Callable, Callable]]
) -> None:
    mlflow.set_experiment(experiment_name)

    args = common.arg_parser().parse_args()
    
    train, val, _ = common.load_data(args.path, args.epochs, args.window_size, args.batch_size)

    checkpoint_manager = tff.simulation.FileCheckpointManager(args.output)

    client_idx2id = list(train.clients.union(val.clients))
    client_id2idx = {i: idx for idx, i in enumerate(client_idx2id)}
    client_states, iterator, validator, evaluator = setup_fn(
        args.window_size,
        args.batch_size,
        args.hidden_size,
        args.dropout_rate,
        args.learning_rate,
        args.pos_weight,
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

        state, client_states = reduce(
            fitting_fn,
            range(1, args.rounds + 1),
            (iterator.initialize(), client_states)
        )

        evaluation_fn(state.model, client_states)
