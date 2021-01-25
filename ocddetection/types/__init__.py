from collections import namedtuple
from typing import Dict, TypeVar

import tensorflow as tf


Metrics = Dict[str, tf.Tensor]

ServerState = TypeVar('ServerState')
ClientState = TypeVar('ClientState')

FederatedDataset = namedtuple('FederatedDataset', ['clients', 'data'])
