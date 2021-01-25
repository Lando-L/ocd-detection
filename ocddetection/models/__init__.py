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
        tf.keras.layers.Activation('relu', name='block_1_activation'),

        tf.keras.layers.Dropout(dropout_rate, name='block_2_dropout'),
        tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(
                hidden_size,
                return_sequences=True,
                dropout=dropout_rate
            ),
            name='block_2_bidirectional'
        ),

        tf.keras.layers.Dropout(dropout_rate, name='block_3_dropout'),
        tf.keras.layers.Dense(output_size, name='block_3_dense')
    ])


def personalized_bidirectional(
    window_size: int,
    feature_size: int,
    output_size: int,
    hidden_size: int,
    dropout_rate: float
):
    base_input = tf.keras.layers.Input((window_size, feature_size), name='base_inputs')
    base_dropout = tf.keras.layers.Dropout(dropout_rate, name='base_dropout')(base_input)
    base_dense = tf.keras.layers.Dense(hidden_size, name='base_dense')(base_dropout)
    base_bn = tf.keras.layers.BatchNormalization(name='base_batch_normalization')(base_dense)
    base_activation = tf.keras.layers.Activation('tanh', name='base_activation')(base_bn)
    base_model = tf.keras.Model(inputs=base_input, outputs=base_activation)

    personalized_input = tf.keras.layers.Input((window_size, hidden_size), name='personalized_inputs')
    personalized_lstm = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(hidden_size, return_sequences=True, dropout=dropout_rate), name='personalized_lstm')(personalized_input)
    
    personalized_output_dropout = tf.keras.layers.Dropout(dropout_rate, name='personalized_dropout')(personalized_lstm)
    personalized_output_dense = tf.keras.layers.Dense(hidden_size, name='personalized_dense')(personalized_output_dropout)
    personalized_model = tf.keras.Model(inputs=personalized_input, outputs=personalized_output_dense)

    model = tf.keras.Model(inputs=base_input, outputs=personalized_model(base_bn))

    return base_model, personalized_model, model
