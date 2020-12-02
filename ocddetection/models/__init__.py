import tensorflow as tf


def bidirectional(
    window_size: int,
    feature_size: int,
    output_size: int,
    hidden_size: int,
    dropout_rate: float
) -> tf.keras.Model:
    return tf.keras.Sequential([
        tf.keras.layers.Input((window_size, feature_size), name='inputs'),

        tf.keras.layers.Dropout(dropout_rate, name='block_1_dropout'),
        tf.keras.layers.Dense(hidden_size, name='block_1_dense'),
        tf.keras.layers.BatchNormalization(name='block_1_batch_normalization'),

        tf.keras.layers.Bidirectional(
            tf.keras.layers.GRU(
                hidden_size,
                return_sequences=True,
                dropout=dropout_rate
            ),
            name='block_2_bidirectional'
        ),

        tf.keras.layers.Dropout(dropout_rate, name='block_2_dropout'),
        tf.keras.layers.Dense(output_size, name='block_2_dense'),
        tf.keras.layers.BatchNormalization(name='block_2_batch_normalization')
    ])
