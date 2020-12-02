import attr

import tensorflow as tf
import tensorflow_federated as tff

from ocddetection.federated.personalization.layers import server, utils


@attr.s(eq=False, frozen=True, slots=True)
class State(object):
    """
    Structure for state on the client.
    
    Fields:
        - `client_index`: The client index integer to map the client state back to the database hosting client states in the driver file..
        - `personalized_variables`: Variables of personalizsed layers.
    """

    client_index = attr.ib()
    personalized_variables = attr.ib()


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


def update(
    dataset: tf.data.Dataset,
    state: State,
    message: server.Message,
    model: utils.PersonalizationLayersDecorator,
    optimizer: tf.keras.optimizers.Optimizer
) -> Output:
    tff.utils.assign(model.base_variables, message.base_variables)
    tff.utils.assign(model.personalized_variables, tf.convert_to_tensor(state.personalized_variables))
    
    client_weight = tf.constant(0, dtype=tf.int32)

    for batch in dataset:
        with tf.GradientTape() as tape:
            outputs = model.forward_pass(batch)

        client_weight += outputs.num_examples
        
        optimizer.apply_gradients(
            zip(
                tape.gradient(outputs.loss, model.trainable_variables),
                model.trainable_variables
            )
        )

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
