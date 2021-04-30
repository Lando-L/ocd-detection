from collections import namedtuple
import csv
from functools import partial
from itertools import chain
import os
import re
from typing import Generator, List, Tuple

import pandas as pd
import tensorflow as tf


SENSORS = [ i-1 for i in
    [
        # Right Lower Arm acc & gyro
        64, 65, 66, 67, 68, 69, 

        # Left Lower Arm acc & gyro
        90, 91, 92, 93, 94, 95
    ]
]

MID_LEVEL_COLUMN = 249

MID_LEVEL_LABELS = {
    0: 'Null',
    406516: 'Open Door 1',
    406517: 'Open Door 2',
    404516: 'Close Door 1',
    404517: 'Close Door 2',
    406520: 'Open Fridge',
    404520: 'Close Fridge',
    406505: 'Open Dishwasher',
    404505: 'Close Dishwasher',
    406519: 'Open Drawer 1',
    404519: 'Close Drawer 1',
    406511: 'Open Drawer 2',
    404511: 'Close Drawer 2',
    406508: 'Open Drawer 3',
    404508: 'Close Drawer 3',
    408512: 'Clean Table',
    407521: 'Drink from Cup',
    405506: 'Toggle Switch'
}
