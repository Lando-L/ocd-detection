import numpy as np
import tensorflow as tf


def bidirectional(window_size: int, feature_size: int, hidden_size: int, pos_weight: int) -> tf.keras.Model:
	return tf.keras.Sequential([
		tf.keras.layers.Input((window_size, feature_size), name='inputs'),

		tf.keras.layers.Bidirectional(
			tf.keras.layers.LSTM(hidden_size, return_sequences=True),
			name='block_1_bidirectional'
		),

		tf.keras.layers.Bidirectional(
			tf.keras.layers.LSTM(hidden_size),
			name='block_2_bidirectional'
		),

		tf.keras.layers.Dense(
			1,
			name='block_3_dense',
			bias_initializer=tf.keras.initializers.Constant(np.log(1 / pos_weight))
		)
	])


def personalized_bidirectional(window_size: int, feature_size: int, hidden_size: int, pos_weight: int) -> tf.keras.Model:
	base_input = tf.keras.layers.Input((window_size, feature_size), name='base_inputs')
	base_block_1_bidirectional = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(hidden_size, return_sequences=True), name='block_1_bidirectional')(base_input)
	base_block_2_bidirectional = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(hidden_size), name='block_2_bidirectional')(base_block_1_bidirectional)
	base_model = tf.keras.Model(inputs=base_input, outputs=base_block_2_bidirectional)

	personalized_input = tf.keras.layers.Input((window_size, hidden_size), name='personalized_inputs')
	personalized_dense = tf.keras.layers.Dense(1, name='personalized_dense', bias_initializer=tf.keras.initializers.Constant(np.log(1 / pos_weight)))(personalized_input)
	personalized_model = tf.keras.Model(inputs=personalized_input, outputs=personalized_dense)

	model = tf.keras.Model(inputs=base_input, outputs=personalized_model(base_block_2_bidirectional))

	return base_model, personalized_model, model
