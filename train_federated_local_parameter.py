import tensorflow as tf
import tensorflow_federated as tff

from ocddetection.federated.learning import stateful
from ocddetection.federated.learning.impl.personalization import layers


def main() -> None:
    tff.backends.native.set_local_execution_context(
        server_tf_device=tf.config.list_logical_devices('CPU')[0],
        client_tf_devices=tf.config.list_logical_devices('GPU')
    )

    stateful.run(
        'Human Activity Recognition',
        'Federated Learning with Personalized Layers',
        layers.setup
    )


if __name__ == "__main__":
    main()
