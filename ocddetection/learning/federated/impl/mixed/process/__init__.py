from typing import Callable, List

import tensorflow as tf
import tensorflow_federated as tff

from ocddetection.learning.federated.impl.mixed import client, server


COEFFICIENT_FN = Callable[[], List[tf.Variable]]
MODEL_FN = Callable[[], tff.learning.Model]
OPTIMIZER_FN = Callable[[], tf.keras.optimizers.Optimizer]
CLIENT_STATE_FN = Callable[[], client.State]
CLIENT_UPDATE_FN = Callable[[tf.data.Dataset, client.State, server.Message, List[tf.Variable], tff.learning.Model, tff.learning.Model, tff.learning.Model, tf.keras.optimizers.Optimizer], client.Output]
SERVER_UPDATE_FN = Callable[[tff.learning.Model, tf.keras.optimizers.Optimizer, server.State, list], server.State]
TRANSFORMATION_FN = Callable[[server.State], server.Message]
VALIDATION_FN = Callable[[tf.data.Dataset, client.State, tff.learning.ModelWeights, tff.learning.Model], client.Validation]
EVALUATION_FN = Callable[[tf.data.Dataset, client.State, tff.learning.ModelWeights, tff.learning.Model], client.Evaluation]


def __initialize_optimizer(
    model: tff.learning.Model,
    optimizer: tf.keras.optimizers.Optimizer
):
    zero_gradient = tf.nest.map_structure(tf.zeros_like, model.trainable_variables)
    optimizer.apply_gradients(zip(zero_gradient, model.trainable_variables))
    assert optimizer.variables()


def __initialize_server(
    model_fn: MODEL_FN,
    optimizer_fn: OPTIMIZER_FN
) -> server.State:
    model = model_fn()
    optimizer = optimizer_fn()
        
    __initialize_optimizer(model, optimizer)

    return server.State(
        model=tff.learning.ModelWeights.from_model(model),
        optimizer_state=optimizer.variables(),
        round_num=0
    )


def __update_server(
    state: server.State,
    weights_delta: list,
    model_fn: MODEL_FN,
    optimizer_fn: OPTIMIZER_FN,
    update_fn: SERVER_UPDATE_FN
) -> server.State:
    model = model_fn()
    optimizer = optimizer_fn()

    __initialize_optimizer(model, optimizer)

    return update_fn(
        model,
        optimizer,
        state,
        weights_delta
    )


def __update_client(
    dataset: tf.data.Dataset,
    state: client.State,
    message: server.Message,
    coefficient_fn: COEFFICIENT_FN,
    model_fn: MODEL_FN,
    optimizer_fn: OPTIMIZER_FN,
    update_fn: CLIENT_UPDATE_FN
) -> client.Output:
    return update_fn(
        dataset,
        state,
        message,
        coefficient_fn,
        model_fn,
        optimizer_fn
    )


def __state_to_message(
    state: server.State,
    transformation_fn: TRANSFORMATION_FN
) -> server.Message:
    return transformation_fn(state)


def iterator(
    coefficient_fn: COEFFICIENT_FN,
    model_fn: MODEL_FN,
    client_state_fn: CLIENT_STATE_FN,
    server_optimizer_fn: OPTIMIZER_FN,
    client_optimizer_fn: OPTIMIZER_FN
):
    model = model_fn()
    client_state = client_state_fn()
    
    init_tf = tff.tf_computation(
        lambda: __initialize_server(
            model_fn,
            server_optimizer_fn
        )
    )
    
    server_state_type = init_tf.type_signature.result
    client_state_type = tff.framework.type_from_tensors(client_state)

    update_server_tf = tff.tf_computation(
        lambda state, weights_delta: __update_server(
            state,
            weights_delta,
            model_fn,
            server_optimizer_fn,
            tf.function(server.update)
        ),
        (server_state_type, server_state_type.model.trainable)
    )

    state_to_message_tf = tff.tf_computation(
        lambda state: __state_to_message(
            state,
            tf.function(server.to_message)
        ),
        server_state_type
    )

    dataset_type = tff.SequenceType(model.input_spec)
    server_message_type = state_to_message_tf.type_signature.result
    
    update_client_tf = tff.tf_computation(
        lambda dataset, state, message: __update_client(
            dataset,
            state,
            message,
            coefficient_fn,
            model_fn,
            client_optimizer_fn,
            tf.function(client.update)
        ),
        (dataset_type, client_state_type, server_message_type)
    )
    
    federated_server_state_type = tff.type_at_server(server_state_type)
    federated_dataset_type = tff.type_at_clients(dataset_type)
    federated_client_state_type = tff.type_at_clients(client_state_type)
    
    def init_tff():
        return tff.federated_value(init_tf(), tff.SERVER)
    
    def next_tff(server_state, datasets, client_states):
        message = tff.federated_map(state_to_message_tf, server_state)
        broadcast = tff.federated_broadcast(message)

        outputs = tff.federated_map(update_client_tf, (datasets, client_states, broadcast))
        weights_delta = tff.federated_mean(outputs.weights_delta, weight=outputs.client_weight)

        metrics = model.federated_output_computation(outputs.metrics)

        next_state = tff.federated_map(update_server_tf, (server_state, weights_delta))

        return next_state, metrics, outputs.client_state
    
    return tff.templates.IterativeProcess(
        initialize_fn=tff.federated_computation(init_tff),
        next_fn=tff.federated_computation(
            next_tff,
            (federated_server_state_type, federated_dataset_type, federated_client_state_type)
        )
    )


def __validate_client(
    dataset: tf.data.Dataset,
    state: client.State,
    weights: tff.learning.ModelWeights,
    coefficient_fn: COEFFICIENT_FN,
    model_fn: MODEL_FN,
    validation_fn: VALIDATION_FN
) -> client.Validation:
    return validation_fn(
        dataset,
        state,
        weights,
        coefficient_fn,
        model_fn
    )


def validator(
    coefficient_fn: COEFFICIENT_FN,
    model_fn: MODEL_FN,
    client_state_fn: CLIENT_STATE_FN
):
    model = model_fn()
    client_state = client_state_fn()

    dataset_type = tff.SequenceType(model.input_spec)
    client_state_type = tff.framework.type_from_tensors(client_state)
    weights_type = tff.learning.framework.weights_type_from_model(model)

    validate_client_tf = tff.tf_computation(
        lambda dataset, state, weights: __validate_client(
            dataset,
            state,
            weights,
            coefficient_fn,
            model_fn,
            tf.function(client.validate)
        ),
        (dataset_type, client_state_type, weights_type)
    )

    federated_weights_type = tff.type_at_server(weights_type)
    federated_dataset_type = tff.type_at_clients(dataset_type)
    federated_client_state_type = tff.type_at_clients(client_state_type)    

    def validate(weights, datasets, client_states):
        broadcast = tff.federated_broadcast(weights)
        outputs = tff.federated_map(validate_client_tf, (datasets, client_states, broadcast))
        metrics = model.federated_output_computation(outputs.metrics)

        return metrics

    return tff.federated_computation(
        validate,
        (federated_weights_type, federated_dataset_type, federated_client_state_type)
    )


def __evaluate_client(
    dataset: tf.data.Dataset,
    state: client.State,
    weights: tff.learning.ModelWeights,
    coefficient_fn: COEFFICIENT_FN,
    model_fn: MODEL_FN,
    evaluation_fn: EVALUATION_FN
) -> client.Evaluation:
    return evaluation_fn(
        dataset,
        state,
        weights,
        coefficient_fn,
        model_fn
    )


def evaluator(
    coefficient_fn: COEFFICIENT_FN,
    model_fn: MODEL_FN,
    client_state_fn: CLIENT_STATE_FN
):
    model = model_fn()
    client_state = client_state_fn()

    dataset_type = tff.SequenceType(model.input_spec)
    client_state_type = tff.framework.type_from_tensors(client_state)
    weights_type = tff.framework.type_from_tensors(tff.learning.ModelWeights.from_model(model))

    evaluate_client_tf = tff.tf_computation(
        lambda dataset, state, weights: __evaluate_client(
            dataset,
            state,
            weights,
            coefficient_fn,
            model_fn,
            tf.function(client.evaluate)
        ),
        (dataset_type, client_state_type, weights_type)
    )

    federated_weights_type = tff.type_at_server(weights_type)
    federated_dataset_type = tff.type_at_clients(dataset_type)
    federated_client_state_type = tff.type_at_clients(client_state_type)    

    def evaluate(weights, datasets, client_states):
        broadcast = tff.federated_broadcast(weights)
        outputs = tff.federated_map(evaluate_client_tf, (datasets, client_states, broadcast))
        
        confusion_matrix = tff.federated_sum(outputs.confusion_matrix)
        aggregated_metrics = model.federated_output_computation(outputs.metrics)
        collected_metrics = tff.federated_collect(outputs.metrics)

        return confusion_matrix, aggregated_metrics, collected_metrics

    return tff.federated_computation(
        evaluate,
        (federated_weights_type, federated_dataset_type, federated_client_state_type)
    )
