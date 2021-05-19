from typing import Callable

import tensorflow as tf
import tensorflow_federated as tff

from ocddetection.learning.federated.stateless.impl.averaging import client, server


MODEL_FN = Callable[[], tff.learning.Model]
OPTIMIZER_FN = Callable[[], tf.keras.optimizers.Optimizer]
CLIENT_UPDATE_FN = Callable[[tf.data.Dataset, server.Message, tff.learning.Model, tf.keras.optimizers.Optimizer], client.Output]
SERVER_UPDATE_FN = Callable[[tff.learning.Model, tf.keras.optimizers.Optimizer, server.State, list], server.State]
TRANSFORMATION_FN = Callable[[server.State], server.Message]
VALIDATION_FN = Callable[[tf.data.Dataset, tff.learning.ModelWeights, tff.learning.Model], client.Validation]
EVALUATION_FN = Callable[[tf.data.Dataset, tff.learning.ModelWeights, tff.learning.Model], client.Evaluation]


def __initialize_optimizer(
    model: tff.learning.Model,
    optimizer: tf.keras.optimizers.Optimizer
):
    model_delta = tf.nest.map_structure(tf.zeros_like, model.trainable_variables)

    optimizer.apply_gradients(
        tf.nest.map_structure(
            lambda x, v: (tf.zeros_like(x), v),
            tf.nest.flatten(model_delta),
            tf.nest.flatten(model.trainable_variables)
        )
    )
    
    assert optimizer.variables()


def __initialize_server(
    model_fn: MODEL_FN,
    optimizer_fn: OPTIMIZER_FN
):
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
):
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
    message: server.Message,
    model_fn: MODEL_FN,
    optimizer_fn: OPTIMIZER_FN,
    update_fn: CLIENT_UPDATE_FN
) -> client.Output:
    return update_fn(
        dataset,
        message,
        model_fn(),
        optimizer_fn()
    )


def __state_to_message(
    state: server.State,
    transformation_fn: TRANSFORMATION_FN
) -> server.Message:
    return transformation_fn(state)


def iterator(
    model_fn: MODEL_FN,
    server_optimizer_fn: OPTIMIZER_FN,
    client_optimizer_fn: OPTIMIZER_FN
):
    model = model_fn()
    
    init_tf = tff.tf_computation(
        lambda: __initialize_server(
            model_fn,
            server_optimizer_fn
        )
    )
    
    server_state_type = init_tf.type_signature.result

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
        lambda dataset, message: __update_client(
            dataset,
            message,
            model_fn,
            client_optimizer_fn,
            tf.function(client.update)
        ),
        (dataset_type, server_message_type)
    )
    
    federated_server_state_type = tff.type_at_server(server_state_type)
    federated_dataset_type = tff.type_at_clients(dataset_type)
    
    def init_tff():
        return tff.federated_value(init_tf(), tff.SERVER)
    
    def next_tff(server_state, datasets):
        message = tff.federated_map(state_to_message_tf, server_state)
        broadcast = tff.federated_broadcast(message)

        outputs = tff.federated_map(update_client_tf, (datasets, broadcast))
        weights_delta = tff.federated_mean(outputs.weights_delta, weight=outputs.client_weight)
        metrics = model.federated_output_computation(outputs.metrics)

        next_state = tff.federated_map(update_server_tf, (server_state, weights_delta))

        return next_state, metrics
    
    return tff.templates.IterativeProcess(
        initialize_fn=tff.federated_computation(init_tff),
        next_fn=tff.federated_computation(
            next_tff,
            (federated_server_state_type, federated_dataset_type)
        )
    )


def __validate_client(
    dataset: tf.data.Dataset,
    weights: tff.learning.ModelWeights,
    model_fn: MODEL_FN,
    validation_fn: VALIDATION_FN
) -> client.Validation:
    return validation_fn(
        dataset,
        weights,
        model_fn()
    )


def validator(model_fn: MODEL_FN):
    model = model_fn()

    dataset_type = tff.SequenceType(model.input_spec)
    weights_type = tff.framework.type_from_tensors(tff.learning.ModelWeights.from_model(model))

    validate_client_tf = tff.tf_computation(
        lambda dataset, weights: __validate_client(
            dataset,
            weights,
            model_fn,
            tf.function(client.validate)
        ),
        (dataset_type, weights_type)
    )

    federated_weights_type = tff.type_at_server(weights_type)
    federated_dataset_type = tff.type_at_clients(dataset_type)   

    def validate(weights, datasets):
        broadcast = tff.federated_broadcast(weights)
        outputs = tff.federated_map(validate_client_tf, (datasets, broadcast))
        metrics = model.federated_output_computation(outputs.metrics)

        return metrics

    return tff.federated_computation(
        validate,
        (federated_weights_type, federated_dataset_type)
    )


def __evaluate_client(
    dataset: tf.data.Dataset,
    weights: tff.learning.ModelWeights,
    model_fn: MODEL_FN,
    evaluation_fn: EVALUATION_FN
) -> client.Evaluation:
    return evaluation_fn(
        dataset,
        weights,
        model_fn()
    )


def evaluator(model_fn: MODEL_FN):
    model = model_fn()

    dataset_type = tff.SequenceType(model.input_spec)
    weights_type = tff.framework.type_from_tensors(tff.learning.ModelWeights.from_model(model))

    evaluate_client_tf = tff.tf_computation(
        lambda dataset, weights: __evaluate_client(
            dataset,
            weights,
            model_fn,
            tf.function(client.evaluate)
        ),
        (dataset_type, weights_type)
    )

    federated_weights_type = tff.type_at_server(weights_type)
    federated_dataset_type = tff.type_at_clients(dataset_type)   

    def evaluate(weights, datasets):
        broadcast = tff.federated_broadcast(weights)
        outputs = tff.federated_map(evaluate_client_tf, (datasets, broadcast))
        
        confusion_matrix = tff.federated_sum(outputs.confusion_matrix)
        metrics = model.federated_output_computation(outputs.metrics)

        return confusion_matrix, metrics

    return tff.federated_computation(
        evaluate,
        (federated_weights_type, federated_dataset_type)
    )
