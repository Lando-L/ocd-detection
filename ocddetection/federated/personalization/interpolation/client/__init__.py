import attr

import tensorflow as tf
import tensorflow_federated as tff

from ocddetection.federated.personalization.interpolation import server


@attr.s(eq=False, frozen=True, slots=True)
class State(object):
    """
    Structure for state on the client.
    
    Fields:
        - `client_index`: The client index integer to map the client state back to the database hosting client states in the driver file..
        - `trainable_variables`: A dictionary of model's trainable variables.
        - `optimizer_state`: Variables of optimizer.
        - `mixing_coefficient`: Mixing coefficient.
    """

    client_index = attr.ib()
    trainable_variables = attr.ib()
    optimizer_state = attr.ib()
    mixing_coefficient = attr.ib()


@attr.s(eq=False, frozen=True, slots=True)
class Output(object):
    """
    Structure for outputs returned from clients during federated optimization.
    
    Fields:
        - `weights_delta`: A dictionary of updates to the model's trainable variables.
        - `client_weight`: Weight to be used in a weighted mean when aggregating `weights_delta`
        - `metrics`: A structure matching `tff.learning.Model.report_local_outputs`, reflecting the results of training on the input dataset.
        - `client_state`: The updated `State`
    """

    weights_delta = attr.ib()
    client_weight = attr.ib()
    metrics = attr.ib()
    client_state = attr.ib()


def __mix_weights(mixing_weight, local_variables, global_variables):
    def get_weights(variables):
        return [variable.numpy() for variable in variables]

    return [
        mixing_weight * local_variable + (1 - mixing_weight) * global_variable
        for local_variable, global_variable in zip(get_weights(local_variables), get_weights(global_variables))
    ]


def update(
    dataset: tf.data.Dataset,
    state: State,
    message: server.Message,
    global_model: tff.learning.Model,
    local_model: tff.learning.Model,
    mixed_model: tff.learning.Model,
    optimizer: tf.keras.optimizers.Optimizer
) -> Output:
    tff.utils.assign(global_model.trainable_variables, message.trainable_variables)
    tff.utils.assign(local_model.trainable_variables, state.trainable_variables)
    tff.utils.assign(
        mixed_model.trainable_variables,
        __mix_weights(
            state.mixing_coefficient,
            state.trainable_variables,
            message.trainable_variables
        )
    )
    
    client_weight = tf.constant(0, dtype=tf.int32)

    for batch in dataset:
        with tf.GradientTape() as tape:
            global_outputs = global_model.forward_pass(batch)
            mixed_outputs = mixed_model.forward_pass(batch)

        client_weight += global_outputs.num_examples
        
        optimizer.apply_gradients(
            zip(
                tape.gradient(global_outputs.loss, global_model.trainable_variables),
                global_model.trainable_variables
            )
        )

        # optimizer.apply_gradients(
        #     zip(
        #         tape.gradient(mixed_outputs.loss, mixed_model.trainable_variables),
        #         local_model.trainable_variables
        #     )
        # )

        # optimizer.apply_gradients(
        #     zip(
        #         tape.gradient(mixed_outputs.loss, local_model.trainable_variables),

        #     )
        # )

    weights_delta = tf.nest.map_structure(
        lambda a, b: a - b,
        model.base_variables,
        message.base_variables
    )
    
    return Output(
        weights_delta=weights_delta,
        client_weight=tf.cast(client_weight, dtype=tf.float32),
        metrics=model.report_local_outputs(),
        client_state=State(
            client_index=state.client_index,
            personalized_variables=model.personalized_variables
        )
    )
