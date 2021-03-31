import attr
import tensorflow as tf
import tensorflow_federated as tff


@attr.s(eq=False, frozen=True, slots=True)
class State(object):
    """
    Structure for state on the server.
    
    Fields:
        - `model`: A ModelWeights structure, containing Tensors or Variables.
        - `optimizer_state`: Variables of optimizer.
        - 'round_num': Current round index
    """

    model = attr.ib()
    optimizer_state = attr.ib()
    round_num = attr.ib()


@attr.s(eq=False, frozen=True, slots=True)
class Message(object):
    """
    Structure for tensors broadcasted by server during federated optimization.
    
    Fields:
        - `model`: A ModelWeights structure, containing Tensors or Variables.
    """

    model = attr.ib()


def update(
    model: tff.learning.Model,
    optimizer: tf.keras.optimizers.Optimizer,
    state: State,
    weights_delta: list
) -> State:
    state.model.assign_weights_to(model)
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
        model=tff.learning.ModelWeights.from_model(model),
        optimizer_state=optimizer.variables(),
        round_num=state.round_num + 1
    )


def to_message(state: State) -> Message:
    return Message(state.model)
