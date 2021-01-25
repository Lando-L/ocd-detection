import mlflow
import tensorflow as tf

class MlFlowLogging(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        mlflow.log_metrics(logs, step=epoch + 1)
