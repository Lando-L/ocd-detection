import attr
from typing import List

import tensorflow as tf
import tensorflow_federated as tff

from ocddetection.learning.federated.stateful.impl.mixed import server


@attr.s(eq=False, frozen=True, slots=True)
class State(object):
    """
    Structure for state on the client.
    
    Fields:
        - `client_index`: The client index integer to map the client state back to the database hosting client states in the driver file..
        - `model`: A ModelWeights structure, containing Tensors or Variables.
        - `mixing_coefficients`: A ModelWeights structure, containing Tensors or Variables.
    """

    client_index = attr.ib()
    model = attr.ib()
    mixing_coefficients = attr.ib()


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


@attr.s(eq=False, frozen=True, slots=True)
class Validation(object):
    """
    Structure for outputs returned from clients during federated validation.
    
    Fields:
        - `metrics`: A structure matching `tff.learning.Model.report_local_outputs`, reflecting the results of training on the input dataset.
    """

    metrics = attr.ib()


@attr.s(eq=False, frozen=True, slots=True)
class Evaluation(object):
    """
    Structure for outputs returned from clients during federated evaluation.
    
    Fields:
        - `confusion_matrix`: Confusion matrix
    """
    confusion_matrix = attr.ib()
    metrics = attr.ib()


def __mix_weights(
    mixing_coefficients: List[tf.Variable],
    local_variables: tff.learning.ModelWeights,
    global_variables: tff.learning.ModelWeights
):
    # alpha * v + (1 - alpha) * w
    def __mix(mixing_coefficients, local_variables, global_variables):
        return [
            tf.add(
                tf.multiply(m, l),
                tf.multiply(
                    tf.subtract(
                        tf.constant(1, dtype=tf.float32),
                        m
                    ),
                    g
                )
            )

            for m, l, g in zip(mixing_coefficients, local_variables, global_variables)
        ]
    
    return tff.learning.ModelWeights(
        trainable=__mix(mixing_coefficients, local_variables.trainable, global_variables.trainable),
        non_trainable=local_variables.non_trainable
    )


def __mixing_gradient(mixed_gradients, local_variables, global_variables):
    # <v - w, f(v, e)>
    def subtract(xs, ys):
        return [
            tf.subtract(x, y)
            for x, y in zip(xs, ys)
        ]

    def inner(xs, ys):
        return [
            tf.tensordot(x, y, axes=len(x.shape))
            for x, y in zip(xs, ys)
        ]
    
    return inner(
        subtract(local_variables, global_variables),
        mixed_gradients
    )


def update(
    dataset: tf.data.Dataset,
    state: State,
    message: server.Message,
    mixing_coefficients: List[tf.Variable],
    global_model: tff.learning.Model,
    local_model: tff.learning.Model,
    mixed_model: tff.learning.Model,
    optimizer: tf.keras.optimizers.Optimizer
) -> Output:
    tff.utils.assign(mixing_coefficients, state.mixing_coefficients)
    message.model.assign_weights_to(global_model)
    state.model.assign_weights_to(local_model)
    __mix_weights(mixing_coefficients, state.model, message.model).assign_weights_to(mixed_model)
    client_weight = tf.constant(0, dtype=tf.int32)

    for batch in dataset:
        with tf.GradientTape() as global_tape:
            global_outputs = global_model.forward_pass(batch, training=True)

        with tf.GradientTape() as mixed_tape:
            mixed_outputs = mixed_model.forward_pass(batch, training=True)

        client_weight += global_outputs.num_examples
        global_gradients = global_tape.gradient(global_outputs.loss, global_model.trainable_variables)
        mixed_gradients = mixed_tape.gradient(mixed_outputs.loss, mixed_model.trainable_variables)
        coefficient_gradient = __mixing_gradient(
            mixed_gradients,
            local_model.trainable_variables,
            global_model.trainable_variables
        )
        
        optimizer.apply_gradients(
            zip(
                global_gradients,
                global_model.trainable_variables
            )
        )
        
        optimizer.apply_gradients(
            zip(
                mixed_gradients,
                local_model.trainable_variables
            )
        )

        optimizer.apply_gradients(
            zip(
                coefficient_gradient,
                mixing_coefficients
            )
        )

        # Clip gradient to be within the interval [0, 1]
        tff.utils.assign(
            mixing_coefficients,
            [tf.clip_by_value(c, 0, 1) for c in mixing_coefficients]
        )

    weights_delta = tf.nest.map_structure(
        lambda a, b: a - b,
        global_model.trainable_variables,
        message.model.trainable
    )
    
    return Output(
        weights_delta=weights_delta,
        client_weight=tf.cast(client_weight, dtype=tf.float32),
        metrics=global_model.report_local_outputs(),
        client_state=State(
            client_index=state.client_index,
            model=tff.learning.ModelWeights.from_model(local_model),
            mixing_coefficients=mixing_coefficients
        )
    )


def validate(
    dataset: tf.data.Dataset,
    state: State,
    weights: tff.learning.ModelWeights,
    mixing_coefficients: List[tf.Variable],
    model: tff.learning.Model
) -> Validation:
    tff.utils.assign(mixing_coefficients, state.mixing_coefficients)
    __mix_weights(mixing_coefficients, state.model, weights).assign_weights_to(model)

    for batch in dataset:
        model.forward_pass(batch, training=False)

    return Validation(
        metrics=model.report_local_outputs(),
    )


def evaluate(
    dataset: tf.data.Dataset,
    state: State,
    weights: tff.learning.ModelWeights,
    mixing_coefficients: List[tf.Variable],
    model: tff.learning.Model
) -> Evaluation:
    tff.utils.assign(mixing_coefficients, state.mixing_coefficients)
    __mix_weights(mixing_coefficients, state.model, weights).assign_weights_to(model)

    def evaluation_fn(state, batch):
        outputs = model.forward_pass(batch, training=False)
        
        y_true = tf.reshape(batch[1], (-1,))
        y_pred = tf.round(tf.nn.sigmoid(tf.reshape(outputs.predictions, (-1,))))

        return state + tf.math.confusion_matrix(y_true, y_pred, num_classes=2)

    confusion_matrix = dataset.reduce(
        tf.zeros((2, 2), dtype=tf.int32),
        evaluation_fn
    )

    return Evaluation(
        confusion_matrix=confusion_matrix,
        metrics=model.report_local_outputs()
    )
