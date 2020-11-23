import numpy as np
import tensorflow as tf


class Bidirectional(tf.keras.Model):
    def __init__(self, hidden_size: int, output_size: int, dropout_rate: float):
        super(Bidirectional, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout_rate = dropout_rate
        
        self.input_dropout = tf.keras.layers.Dropout(dropout_rate)
        self.input_dense = tf.keras.layers.Dense(hidden_size)
        self.input_bn = tf.keras.layers.BatchNormalization()

        self.gru = tf.keras.layers.GRU(hidden_size, return_sequences=True, dropout=dropout_rate)
        self.gru_bidirectional = tf.keras.layers.Bidirectional(self.gru)

        self.output_dropout = tf.keras.layers.Dropout(dropout_rate)
        self.output_dense = tf.keras.layers.Dense(output_size)
        self.output_bn = tf.keras.layers.BatchNormalization()

    def reset_states(self):
        return self.gru_bidirectional.reset_states()

    def call(self, x, training=False):
        x = self.input_dropout(x, training=training)
        x = self.input_dense(x, training=training)
        x = self.input_bn(x, training=training)
    
        x = self.gru_bidirectional(x, training=training)
        
        x = self.output_dropout(x, training=training)
        x = self.output_dense(x, training=training)
        x = self.output_bn(x, training=training)
        
        return x
