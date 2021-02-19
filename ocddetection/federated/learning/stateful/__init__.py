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
    train_fn: Callable[[ServerState, List[tf.data.Dataset], List[ClientState]], Tuple[ServerState, Metrics, List[ClientState]]]
) -> Tuple[ServerState, Metrics, Dict[int, ClientState]]:
    sampled_clients = common.sample_clients(dataset, clients_per_round)
    sampled_data = [dataset.data[client] for client in sampled_clients]
    sampled_client_states = [client_states[client] for client in sampled_clients]
    
    next_state, metrics, updated_client_states = train_fn(server_state, sampled_data, sampled_client_states)
    next_client_states = {
        **client_states,
        **{
            client_idx2id[client.client_index.numpy()]: client
            for client in updated_client_states
        }
    }

    return next_state, metrics, next_client_states


def __evaluation_step(
    weights: tff.learning.ModelWeights,
    client_states: ClientState,
    dataset: FederatedDataset,
    evaluate_fn: Callable[[tff.learning.ModelWeights, List[tf.data.Dataset], List[ClientState]], Metrics]
) -> Tuple[Metrics, List[tf.Tensor]]:
    metrics, analysis = evaluate_fn(
        weights,
        [dataset.data[client] for client in dataset.clients],
        [client_states[client] for client in dataset.clients]
    )

    return common.update_test_metrics(metrics), analysis


def __step(
    evaluation_round_rate: int,
    train_step_fn: Callable[[ServerState, Dict[int, ClientState]], Tuple[ServerState, Metrics, Dict[int, ClientState]]],
    evaluation_step_fn: Callable[[tff.learning.ModelWeights, Dict[int, ClientState]], Metrics]
) -> Callable[[Tuple[ServerState, Dict[int, ClientState]], int], Tuple[ServerState, Dict[int, ClientState]]]:
    def __reduce_fn(aggregates: Tuple[ServerState, Dict[int, ClientState]], round_num: int) -> Tuple[ServerState, Dict[int, ClientState]]:
        state, metrics, client_states = train_step_fn(aggregates[0], aggregates[1])
        mlflow.log_metrics(metrics['train'], step=round_num)

        if round_num % evaluation_round_rate == 0:
            test_metrics, test_analysis = evaluation_step_fn(state.model, client_states)
            mlflow.log_metrics(test_metrics, step=round_num)

            for idx, t in enumerate(test_analysis):
                fig, ax = plt.subplots()
                cm = t / t.numpy().sum(axis=1)[:, tf.newaxis]
                sns.heatmap(cm, annot=True, ax=ax)
                plt.xlabel('Predicted')
                plt.ylabel('True')
                mlflow.log_figure(fig, f'cm_{round_num}_{idx}.png')
                plt.close(fig)

        return state, client_states
    
    return __reduce_fn


def run(
    experiment_name: str,
    run_name: str,
    setup_fn: Callable[[int, int, float, float, Dict[int, int]], Tuple[Dict[int, ClientState], Callable, Callable]]
) -> None:
    mlflow.set_experiment(experiment_name)

    args = common.arg_parser().parse_args()
    
    train, val = common.load_data(args.path, args.epochs, args.window_size, args.batch_size)

    client_idx2id = list(train.clients.union(val.clients))
    client_id2idx = {i: idx for idx, i in enumerate(client_idx2id)}
    client_states, iterator, evaluator = setup_fn(
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

        state, client_states = reduce(
            reduce_fn,
            range(1, args.rounds + 1),
            (
                iterator.initialize(),
                client_states
            )
        )
