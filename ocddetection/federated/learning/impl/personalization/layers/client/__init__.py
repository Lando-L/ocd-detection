import attr

import tensorflow as tf
import tensorflow_federated as tff

from ocddetection.federated.learning.impl.personalization.layers import server, utils


@attr.s(eq=False, frozen=True, slots=True)
class State(object):
    """
    Structure for state on the client.
    
    Fields:
        - `client_index`: The client index integer to map the client state back to the database hosting client states in the driver file..
        - `model`: A ModelWeights structure, containing Tensors or Variables.
    """

    client_index = attr.ib()
    model = attr.ib()


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


def update(
    dataset: tf.data.Dataset,
    state: State,
    message: server.Message,
    model: utils.PersonalizationLayersDecorator,
    optimizer: tf.keras.optimizers.Optimizer
) -> Output:
    message.model.assign_weights_to(model.base_model)
    state.model.assign_weights_to(model.personalized_model)
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
        model.base_model.trainable_variables,
        message.model.trainable
    )
    
    return Output(
        weights_delta=weights_delta,
        client_weight=tf.cast(client_weight, dtype=tf.float32),
        metrics=model.report_local_outputs(),
        client_state=State(
            client_index=state.client_index,
            model=tff.learning.ModelWeights.from_model(model.personalized_model)
        )
    )


def validate(
    dataset: tf.data.Dataset,
    state: State,
    weights: tff.learning.ModelWeights,
    model: utils.PersonalizationLayersDecorator
) -> Validation:
    weights.assign_weights_to(model.base_model)
    state.model.assign_weights_to(model.personalized_model)

    for batch in dataset:
        model.forward_pass(batch, training=False)

    return Validation(
        metrics=model.report_local_outputs(),
    )


def evaluate(
    dataset: tf.data.Dataset,
    state: State,
    weights: tff.learning.ModelWeights,
    model: utils.PersonalizationLayersDecorator
) -> Evaluation:
    weights.assign_weights_to(model.base_model)
    state.model.assign_weights_to(model.personalized_model)

    def __evaluation_fn(state, batch):
        outputs = model.forward_pass(batch, training=False)
        
        y_true = tf.reshape(batch[1], (-1,))
        y_pred = tf.round(tf.nn.sigmoid(tf.reshape(outputs.predictions, (-1,))))

        return tf.math.confusion_matrix(y_true, y_pred, num_classes=2)

    return Evaluation(
        confusion_matrix=dataset.reduce(tf.zeros((2, 2), dtype=tf.int32), __evaluation_fn)
    )
