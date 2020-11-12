from functools import partial
from typing import Tuple

import tensorflow as tf

from ocddetection.models.bidirectional import Bidirectional


def build(hidden_size: int, output_size: int, dropout_rate: float) -> Bidirectional:
    return Bidirectional(hidden_size, output_size, dropout_rate)


def train_step(X, y, fwd_state, bwd_state, model, optimizer, loss_fn, train_loss, train_accuracy):
    with tf.GradientTape() as tape:
        y_hat, fwd_state, bwd_state = model(X, [fwd_state, bwd_state], training=True)
        loss = loss_fn(y, y_hat)

    gradients = tape.gradient(loss, model.trainable_variables)  
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    train_loss(loss)
    train_accuracy(y, y_hat)

    return fwd_state, bwd_state


def test_step(X, y, fwd_state, bwd_state, model, loss_fn, test_loss, test_accuracy):
    y_hat, fwd_state, bwd_state = model(X, [fwd_state, bwd_state])
    loss = loss_fn(y, y_hat)

    test_loss(loss)
    test_accuracy(y, y_hat)

    return fwd_state, bwd_state
