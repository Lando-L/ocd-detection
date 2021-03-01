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
        38, 39, 40, 41, 42, 43, 44, 45, 46, 51, 52, 53, 54, 55, 56, 57, 58, 59,
        64, 65, 66, 67, 68, 69, 70, 71, 72, 77, 78, 79, 80, 81, 82, 83, 84, 85,
        90, 91, 92, 93, 94, 95, 96, 97, 98, 103, 104, 105, 106, 107, 108, 109,
        110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124,
        125, 126, 127, 128, 129, 130, 131, 132, 133, 134
    ]
]

MID_LEVEL = 249

MID_LEVEL_LABELS = [
    (0, 'Null'),
    (406516, 'Open Door 1'),
    (406517, 'Open Door 2'),
    (404516, 'Close Door 1'),
    (404517, 'Close Door 2'),
    (406520, 'Open Fridge'),
    (404520, 'Close Fridge'),
    (406505, 'Open Dishwasher'),
    (404505, 'Close Dishwasher'),
    (406519, 'Open Drawer 1'),
    (404519, 'Close Drawer 1'),
    (406511, 'Open Drawer 2'),
    (404511, 'Close Drawer 2'),
    (406508, 'Open Drawer 3'),
    (404508, 'Close Drawer 3'),
    (408512, 'Clean Table'),
    (407521, 'Drink from Cup'),
    (405506, 'Toggle Switch')
]

LABEL2IDX = {label: i for i, (label, _) in enumerate(MID_LEVEL_LABELS)}
IDX2LABEL = [label for label, _ in MID_LEVEL_LABELS]


def files(path: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    AdlFile = namedtuple('AdlFile', ['subject', 'run', 'path'])

    def adl_files(path: str) -> Generator[AdlFile, None, None]:
        for file_name in os.listdir(path):
            match = re.fullmatch('S([0-9])-ADL([a-zA-Z0-9]+)[.]dat', file_name)
            if match:
                yield AdlFile(int(match.group(1)), int(match.group(2)), os.path.join(path, file_name))
    
    DrillFile = namedtuple('DrillFile', ['subject', 'path'])

    def drill_files(path: str) -> Generator[DrillFile, None, None]:
        for file_name in os.listdir(path):
            match = re.fullmatch('S([0-9])-Drill[.]dat', file_name)
            if match:
                yield DrillFile(int(match.group(1)), os.path.join(path, file_name))
    
    adls = pd.DataFrame.from_records(list(adl_files(path)), columns=AdlFile._fields).set_index(['subject', 'run'])
    drills = pd.DataFrame.from_records(list(drill_files(path)), columns=DrillFile._fields).set_index(['subject'])

    return adls, drills


def to_dataset(paths: List[str]) -> tf.data.Dataset:
    def read(path: str) -> Generator[List[float], None, None]:
        with open(path, 'r', newline='') as file:
            reader = csv.reader(file, delimiter=' ', quoting=csv.QUOTE_NONNUMERIC)
            for row in reader:
                yield row
    
    return tf.data.Dataset.from_generator(
        lambda: chain.from_iterable(map(read, paths)),
        output_types=tf.float32,
        output_shapes=(250)
    )
