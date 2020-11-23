import attr

import tensorflow as tf


@attr.s(eq=False, frozen=True, slots=True)
class ClientOutput(object):
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


def update(dataset, message, model, optimizer):
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
        model.trainable_variables,
        message.trainable_variables
    )
    
    return ClientOutput(
        weights_delta=weights_delta,
        client_weight=tf.cast(client_weight, dtype=tf.float32),
        metrics=model.report_local_outputs()
    )
