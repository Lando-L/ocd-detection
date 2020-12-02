import attr
import tensorflow as tf
import tensorflow_federated as tff

from ocddetection.federated.personalization.layers import utils


@attr.s(eq=False, frozen=True, slots=True)
class State(object):
    """
    Structure for state on the server.
    
    Fields:
        - `base_variables`: A dictionary of model's base variables.
        - `optimizer_state`: Variables of optimizer.
        - 'round_num': Current round index
    """

    base_variables = attr.ib()
    optimizer_state = attr.ib()
    round_num = attr.ib()


@attr.s(eq=False, frozen=True, slots=True)
class Message(object):
    """
    Structure for tensors broadcasted by server during federated optimization.
    
    Fields:
        - `base_variables`: A dictionary of model's base variables.
    """

    base_variables = attr.ib()


def update(
    model: utils.PersonalizationLayersDecorator,
    optimizer: tf.keras.optimizers.Optimizer,
    state: State,
    weights_delta: list
) -> State:
    tff.utils.assign(model.base_variables, state.base_variables)
    tff.utils.assign(optimizer.variables(), state.optimizer_state)

    optimizer.apply_gradients(
        tf.nest.map_structure(
            lambda x, v: (-1.0 * x, v),
            tf.nest.flatten(weights_delta),
            tf.nest.flatten(model.base_variables)
        )
    )

    return tff.utils.update_state(
        state,
        base_variables=model.base_variables,
        optimizer_state=optimizer.variables(),
        round_num=state.round_num + 1
    )


def to_message(state: State) -> Message:
    return Message(
        state.base_variables
    )
