from collections import namedtuple

Config = namedtuple(
  'Config',
  ['path', 'output', 'learning_rate', 'epochs', 'batch_size', 'window_size', 'pos_weight', 'checkpoint_rate', 'hidden_size']
)
