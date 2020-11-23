from functools import partial

import tensorflow as tf
import tensorflow_federated as tff

from ocddetection.federated import client, server


def __initialize_optimizer(model, optimizer):
    model_delta = tf.nest.map_structure(tf.zeros_like, model.trainable_variables)

    optimizer.apply_gradients(
        tf.nest.map_structure(
            lambda x, v: (tf.zeros_like(x), v),
            tf.nest.flatten(model_delta),
            tf.nest.flatten(model.trainable_variables)
        )
    )
    
    assert optimizer.variables()


def __initialize_server(model_fn, optimizer_fn):
    model = model_fn()
    optimizer = optimizer_fn()
    __initialize_optimizer(model, optimizer)

    return server.State(
        trainable_variables=model.trainable_variables,
        optimizer_state=optimizer.variables(),
        round_num=0
    )


def __update_server(state, weights_delta, model_fn, optimizer_fn, update_fn):
    model = model_fn()
    optimizer = optimizer_fn()
    __initialize_optimizer(model, optimizer)

    return update_fn(
        model,
        optimizer,
        state,
        weights_delta
    )


def __update_client(dataset, message, model_fn, optimizer_fn, update_fn):
    model = model_fn()
    optimizer = optimizer_fn()

    return update_fn(
        dataset,
        message,
        model,
        optimizer
    )


def iterator(model_fn, server_optimizer_fn, client_optimizer_fn):
    init_tf = tff.tf_computation(
        lambda: __initialize_server(
            model_fn,
            server_optimizer_fn
        )
    )
    
    model = model_fn()
    server_state_type = init_tf.type_signature.result
    dataset_type = tff.SequenceType(model.input_spec)
    
    update_server_tf = tff.tf_computation(
        lambda state, weights_delta: __update_server(
            state,
            weights_delta,
            model_fn,
            server_optimizer_fn,
            tf.function(server.update)
        ),
        (server_state_type, server_state_type.trainable_variables)
    )
    
    update_client_tf = tff.tf_computation(
        lambda dataset, message: __update_client(
            dataset,
            message,
            model_fn,
            client_optimizer_fn,
            tf.function(client.update)
        ),
        (dataset_type, server_state_type)
    )
    
    federated_server_state_type = tff.type_at_server(server_state_type)
    federated_dataset_type = tff.type_at_clients(dataset_type)
    
    def init_tff():
        return tff.federated_value(init_tf(), tff.SERVER)
    
    def next_tff(state, dataset):
        message = tff.federated_broadcast(state)

        outputs = tff.federated_map(update_client_tf, (dataset, message))
        weights_delta = tff.federated_mean(outputs.weights_delta, weight=outputs.client_weight)
        metrics = model.federated_output_computation(outputs.metrics)

        next_state = tff.federated_map(update_server_tf, (state, weights_delta))

        return next_state, metrics
    
    return tff.templates.IterativeProcess(
        initialize_fn=tff.federated_computation(init_tff),
        next_fn=tff.federated_computation(next_tff, (federated_server_state_type, federated_dataset_type))
    )
