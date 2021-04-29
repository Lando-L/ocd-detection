import tensorflow as tf
import tensorflow_federated as tff

from ocddetection.learning.federated import stateful
from ocddetection.learning.federated.stateful import layers


def main() -> None:
    tff.backends.native.set_local_execution_context(
        server_tf_device=tf.config.list_logical_devices('CPU')[0],
        client_tf_devices=tf.config.list_logical_devices('GPU')
    )

    stateful.run(
        'OCD Detection',
        'Federated Learning with Personalized Layers',
        layers.setup
    )


if __name__ == "__main__":
    main()
