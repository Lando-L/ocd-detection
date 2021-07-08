from typing import Callable
import attr

import tensorflow as tf
import tensorflow_federated as tff

from ocddetection.learning.federated.impl.averaging import server


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
    client_pos_weight = attr.ib()


@attr.s(eq=False, frozen=True, slots=True)
class Output(object):
    """
    Structure for outputs returned from clients during federated optimization.
    
    Fields:
        - `weights_delta`: A dictionary of updates to the model's trainable variables.
        - `client_weight`: Weight to be used in a weighted mean when aggregating `weights_delta`
        - `metrics`: A structure matching `tff.learning.Model.report_local_outputs`, reflecting the results of training on the input dataset.
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
        - `metrics`: A structure matching `tff.learning.Model.report_local_outputs`, reflecting the results of training on the input dataset.
    """
    confusion_matrix = attr.ib()
    metrics = attr.ib()


def update(
    dataset: tf.data.Dataset,
    state: State,
    message: server.Message,
    model_fn: Callable,
    optimizer_fn: Callable
) -> Output:
    with tf.init_scope():
        model = model_fn(pos_weight=state.client_pos_weight)
        optimizer = optimizer_fn()

    message.model.assign_weights_to(model)

    def training_fn(num_examples, batch):
        with tf.GradientTape() as tape:
            outputs = model.forward_pass(batch, training=True)

        optimizer.apply_gradients(
            zip(
                tape.gradient(outputs.loss, model.trainable_variables),
                model.trainable_variables
            )
        )

        return num_examples + outputs.num_examples

    client_weight = dataset.reduce(
        tf.constant(0, dtype=tf.int32),
        training_fn
    )

    weights_delta = tf.nest.map_structure(
        lambda a, b: a - b,
        model.trainable_variables,
        message.model.trainable
    )
    
    return Output(
        weights_delta=weights_delta,
        metrics=model.report_local_outputs(),
        client_weight=tf.cast(client_weight, dtype=tf.float32),
        client_state=State(
            client_index=state.client_index,
            client_pos_weight=state.client_pos_weight
        )
    )


def validate(
    dataset: tf.data.Dataset,
    state: State,
    weights: tff.learning.ModelWeights,
    model_fn: Callable
) -> Validation:
    with tf.init_scope():
        model = model_fn(pos_weight=state.client_pos_weight)

    weights.assign_weights_to(model)

    for batch in dataset:
        model.forward_pass(batch, training=False)

    return Validation(
        metrics=model.report_local_outputs()
    )


def evaluate(
    dataset: tf.data.Dataset,
    state: State,
    weights: tff.learning.ModelWeights,
    model_fn: Callable
) -> Evaluation:
    with tf.init_scope():
        model = model_fn(pos_weight=state.client_pos_weight)
    
    weights.assign_weights_to(model)

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
