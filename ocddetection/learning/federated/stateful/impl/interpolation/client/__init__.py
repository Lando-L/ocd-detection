from typing import Callable
import attr

import tensorflow as tf
import tensorflow_federated as tff

from ocddetection.learning.federated.stateful.impl.interpolation import server


@attr.s(eq=False, frozen=True, slots=True)
class State(object):
    """
    Structure for state on the client.
    
    Fields:
        - `client_index`: The client index integer to map the client state back to the database hosting client states in the driver file..
        - `model`: A ModelWeights structure, containing Tensors or Variables.
        - `mixing_coefficient`: A ModelWeights structure, containing Tensors or Variables.
    """

    client_index = attr.ib()
    model = attr.ib()
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
    mixing_coefficient: tf.Variable,
    local_variables: tff.learning.ModelWeights,
    global_variables: tff.learning.ModelWeights
):
    # alpha * v + (1 - alpha) * w
    def __mix(mixing_coefficient, local_variables, global_variables):
        return [
            tf.add(
                tf.multiply(mixing_coefficient, local_variable),
                tf.multiply(
                    tf.subtract(
                        tf.constant(1, dtype=tf.float32),
                        mixing_coefficient
                    ),
                    global_variable
                )
            )

            for local_variable, global_variable in zip(local_variables,global_variables)
        ]
    
    return tff.learning.ModelWeights(
        trainable=__mix(mixing_coefficient, local_variables.trainable, global_variables.trainable),
        non_trainable=__mix(mixing_coefficient, local_variables.non_trainable, global_variables.non_trainable)
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
    
    return tf.reduce_mean(
        tf.stack(
            inner(
                subtract(local_variables, global_variables),
                mixed_gradients
            ),
            axis=0
        )
    )


def update(
    dataset: tf.data.Dataset,
    state: State,
    message: server.Message,
    coefficient_fn: Callable,
    model_fn: Callable,
    optimizer_fn: Callable
) -> Output:
    with tf.init_scope():
        mixing_optimizer = optimizer_fn()
        mixing_coefficient = coefficient_fn()
        
        global_optimizer = optimizer_fn()
        global_model = model_fn()
        
        local_optimizer = optimizer_fn()
        local_model = model_fn()

        mixed_model = model_fn()

    mixing_coefficient.assign(state.mixing_coefficient)
    message.model.assign_weights_to(global_model)
    state.model.assign_weights_to(local_model)
    __mix_weights(mixing_coefficient, state.model, message.model).assign_weights_to(mixed_model)
    client_weight = tf.constant(0, dtype=tf.int32)

    for batch in dataset:
        with tf.GradientTape() as global_tape:
            global_outputs = global_model.forward_pass(batch, training=True)

        with tf.GradientTape() as mixed_tape:
            mixed_outputs = mixed_model.forward_pass(batch, training=True)

        client_weight += global_outputs.num_examples
        global_gradients = global_tape.gradient(global_outputs.loss, global_model.trainable_variables)
        mixed_gradients = mixed_tape.gradient(mixed_outputs.loss, mixed_model.trainable_variables)
        coefficient_gradient = __mixing_gradient(mixed_gradients, local_model.trainable_variables, global_model.trainable_variables)
        
        # Update global model
        global_optimizer.apply_gradients(
            zip(
                global_gradients,
                global_model.trainable_variables
            )
        )
        
        # Update local model
        local_optimizer.apply_gradients(
            zip(
                mixed_gradients,
                local_model.trainable_variables
            )
        )

        # Update coefficient 
        mixing_optimizer.apply_gradients(
            [(coefficient_gradient, mixing_coefficient)]
        )

        # Clip gradient to be within the interval [0, 1]
        mixing_coefficient.assign(
            tf.clip_by_value(mixing_coefficient, 0, 1)
        )

        __mix_weights(
            mixing_coefficient,
            tff.learning.ModelWeights.from_model(local_model),
            tff.learning.ModelWeights.from_model(global_model)
        ).assign_weights_to(mixed_model)

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
            mixing_coefficient=mixing_coefficient
        )
    )


def validate(
    dataset: tf.data.Dataset,
    state: State,
    weights: tff.learning.ModelWeights,
    coefficient_fn: Callable,
    model_fn: Callable
) -> Validation:
    with tf.init_scope():
        mixing_coefficient = coefficient_fn()
        model = model_fn()

    mixing_coefficient.assign(state.mixing_coefficient)
    __mix_weights(mixing_coefficient, state.model, weights).assign_weights_to(model)

    for batch in dataset:
        model.forward_pass(batch, training=False)

    return Validation(
        metrics=model.report_local_outputs(),
    )


def evaluate(
    dataset: tf.data.Dataset,
    state: State,
    weights: tff.learning.ModelWeights,
    coefficient_fn: Callable,
    model_fn: Callable
) -> Evaluation:
    with tf.init_scope():
        mixing_coefficient = coefficient_fn()
        model = model_fn()

    mixing_coefficient.assign(state.mixing_coefficient)
    __mix_weights(mixing_coefficient, state.model, weights).assign_weights_to(model)

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
