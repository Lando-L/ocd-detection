import attr
import tensorflow as tf
import tensorflow_federated as tff


@attr.s(eq=False, frozen=True, slots=True)
class State(object):
    """
    Structure for state on the server.
    
    Fields:
        - `trainable_variables`: A dictionary of model's trainable variables.
        - `optimizer_state`: Variables of optimizer.
        - 'round_num': Current round index
    """

    trainable_variables = attr.ib()
    optimizer_state = attr.ib()
    round_num = attr.ib()


@attr.s(eq=False, frozen=True, slots=True)
class Message(object):
    """
    Structure for tensors broadcasted by server during federated optimization.
    
    Fields:
        - `trainable_variables`: A dictionary of model's trainable variables.
    """

    trainable_variables = attr.ib()


def update(
    model: tff.learning.Model,
    optimizer: tf.keras.optimizers.Optimizer,
    state: State,
    weights_delta: list
) -> State:
    tff.utils.assign(model.trainable_variables, state.trainable_variables)
    tff.utils.assign(optimizer.variables(), state.optimizer_state)

    optimizer.apply_gradients(
        tf.nest.map_structure(
            lambda x, v: (-1.0 * x, v),
            tf.nest.flatten(weights_delta),
            tf.nest.flatten(model.trainable_variables)
        )
    )

    return tff.utils.update_state(
        state,
        trainable_variables=model.trainable_variables,
        optimizer_state=optimizer.variables(),
        round_num=state.round_num + 1
    )

def to_message(state: State) -> Message:
    return Message(
        state.trainable_variables
    )
