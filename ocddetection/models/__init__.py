import numpy as np
import tensorflow as tf


def bidirectional(
    window_size: int,
    feature_size: int,
    hidden_size: int,
    dropout_rate: float
) -> tf.keras.Model:
    return tf.keras.Sequential([
        #Input
        tf.keras.layers.Input((window_size, feature_size), name='inputs'),

        # Dense Block
        # tf.keras.layers.Dropout(dropout_rate, name='block_1_dropout'),
        # tf.keras.layers.Dense(hidden_size, name='block_1_dense', kernel_constraint=tf.keras.constraints.min_max_norm(0.0, 2.0)),
        # tf.keras.layers.BatchNormalization(name='block_1_bn'),
        # tf.keras.layers.Activation('relu', name='block_1_activation'),

        tf.keras.layers.Dense(hidden_size, name='block_1_dense'),
        tf.keras.layers.Activation('relu', name='block_1_activation'),

        # Bidirectional Recurrent Block
        # tf.keras.layers.Dropout(dropout_rate, name='block_2_dropout'),
        tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(
                hidden_size,
                # return_sequences=True,
                # dropout=dropout_rate,
                # kernel_constraint=tf.keras.constraints.min_max_norm(0.0, 2.0),
                stateful=False
            ),
            name='block_2_bidirectional'
        ),

        # Output Dense Block
        # tf.keras.layers.Dropout(dropout_rate, name='block_3_dropout'),
        tf.keras.layers.Dense(
            1,
            name='block_3_dense'
            # kernel_constraint=tf.keras.constraints.min_max_norm(0.0, 2.0),
            # bias_initializer=output_bias_initializer
        )
    ])


def personalized_bidirectional(
    window_size: int,
    feature_size: int,
    hidden_size: int,
    dropout_rate: float
):
    base_input = tf.keras.layers.Input((window_size, feature_size), name='base_inputs')
    base_dense = tf.keras.layers.Dense(hidden_size, name='base_dense')(base_input)
    base_activation = tf.keras.layers.Activation('relu', name='base_activation')(base_dense)
    base_model = tf.keras.Model(inputs=base_input, outputs=base_activation)

    personalized_input = tf.keras.layers.Input((window_size, hidden_size), name='personalized_inputs')
    personalized_lstm = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(hidden_size, name='personalized_lstm'))(personalized_input)
    personalized_dense = tf.keras.layers.Dense(1, name='personalized_dense')(personalized_lstm)
    personalized_model = tf.keras.Model(inputs=personalized_input, outputs=personalized_dense)

    model = tf.keras.Model(inputs=base_input, outputs=personalized_model(base_activation))

    return base_model, personalized_model, model
