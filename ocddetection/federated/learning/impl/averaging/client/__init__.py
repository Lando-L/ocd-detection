import attr

import tensorflow as tf
import tensorflow_federated as tff

from ocddetection.federated.learning.impl.averaging import server


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


def update(
    dataset: tf.data.Dataset,
    message: server.Message,
    model: tff.learning.Model,
    optimizer: tf.keras.optimizers.Optimizer
) -> Output:
    message.model.assign_weights_to(model)
    client_weight = tf.constant(0, dtype=tf.int32)

    for batch in dataset:
        with tf.GradientTape() as tape:
            outputs = model.forward_pass(batch, training=True)

        client_weight += outputs.num_examples
        
        optimizer.apply_gradients(
            zip(
                tape.gradient(outputs.loss, model.trainable_variables),
                model.trainable_variables
            )
        )

    weights_delta = tf.nest.map_structure(
        lambda a, b: a - b,
        model.trainable_variables,
        message.model.trainable_variables
    )
    
    return Output(
        weights_delta=weights_delta,
        client_weight=tf.cast(client_weight, dtype=tf.float32),
        metrics=model.report_local_outputs()
    )


def validate(
    dataset: tf.data.Dataset,
    weights: tff.learning.ModelWeights,
    model: tff.learning.Model
) -> Validation:
    weights.assign_weights_to(model)

    for batch in dataset:
        model.forward_pass(batch, training=False)

    return Validation(
        metrics=model.report_local_outputs(),
    )


def evaluate(
    dataset: tf.data.Dataset,
    weights: tff.learning.ModelWeights,
    model: tff.learning.Model
) -> Evaluation:
    weights.assign_weights_to(model)

    def __evaluation_fn(state, batch):
        outputs = model.forward_pass(batch, training=False)
        
        y_true = tf.reshape(batch[1], (-1,))
        y_pred = tf.round(tf.nn.sigmoid(tf.reshape(outputs.predictions, (-1,))))

        return tf.math.confusion_matrix(y_true, y_pred, num_classes=2)

    return Evaluation(
        confusion_matrix=dataset.reduce(tf.zeros((2, 2), dtype=tf.int32), __evaluation_fn)
    )
