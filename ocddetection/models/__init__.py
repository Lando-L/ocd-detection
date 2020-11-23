from functools import partial
from typing import Tuple

import tensorflow as tf

from ocddetection.models.bidirectional import Bidirectional


def build(input_shape: Tuple, batch_size: int, output_size: int, hidden_size: int, dropout_rate: float):
    return tf.keras.Sequential([
        tf.keras.layers.Input(input_shape, batch_size, name='inputs'),
        
        tf.keras.layers.Dropout(dropout_rate, name='block_1_dropout'),
        tf.keras.layers.Dense(hidden_size, name='block_1_dense'),
        tf.keras.layers.BatchNormalization(name='block_1_batch_normalization'),

        tf.keras.layers.Bidirectional(
            tf.keras.layers.GRU(
                hidden_size,
                return_sequences=True,
                dropout=dropout_rate,
                stateful=True
            ),
            name='block_2_bidirectional'
        ),

        tf.keras.layers.Dropout(dropout_rate, name='block_2_dropout'),
        tf.keras.layers.Dense(output_size, name='block_2_dense'),
        tf.keras.layers.BatchNormalization(name='block_2_batch_normalization')
    ])


def train_step(X, y, model, optimizer, loss_fn, output_size, train_loss, train_mean_f1_score, train_weighted_f1_score):
    with tf.GradientTape() as tape:
        y_hat = model(X, training=True)
        loss = loss_fn(y, y_hat)

    gradients = tape.gradient(loss, model.trainable_variables)  
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    # Reshape for metrics
    _y = tf.reshape(tf.one_hot(y, depth=output_size), [-1, output_size])
    _y_hat = tf.reshape(y_hat, [-1, output_size])

    train_loss(loss)
    train_mean_f1_score(_y, _y_hat)
    train_weighted_f1_score(_y, _y_hat)


def test_step(X, y, model, loss_fn, output_size, test_loss, test_mean_f1_score, test_weighted_f1_score):
    y_hat = model(X, training=False)
    loss = loss_fn(y, y_hat)

    # Reshape for metrics
    _y = tf.reshape(tf.one_hot(y, depth=output_size), [-1, output_size])
    _y_hat = tf.reshape(y_hat, [-1, output_size])

    test_loss(loss)
    test_mean_f1_score(_y, _y_hat)
    test_weighted_f1_score(_y, _y_hat)
