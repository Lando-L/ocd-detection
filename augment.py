from argparse import ArgumentParser
from functools import partial
import glob
import os
import re
from typing import List, Text, Tuple

import pandas as pd

from ocddetection import data
from ocddetection.data import augmentation


def __arg_parser() -> ArgumentParser:
    parser = ArgumentParser()
    
    parser.add_argument('path', type=str)
    parser.add_argument('output', type=str)

    parser.add_argument('--include-original', dest='include_original', action='store_const', const=True, default=True)
    parser.add_argument('--exclude-original', dest='include_original', action='store_const', const=False, default=True)

    return parser


def __get_files(pattern, root: Text) -> Tuple[List[Text], List[pd.DataFrame]]:
    ids = []
    files = []
    
    for path in glob.iglob(root):
        matched = pattern.match(os.path.basename(path))
        
        if matched:
            ids.append(int(matched.group(2)))
            files.append(augmentation.read_dat(path))
    
    return ids, files


def __write_augmented(df: pd.DataFrame, output: Text, subject: int, run: int) -> None:
    df.drop(data.MID_LEVEL_COLUMN, axis=1).to_csv(
        os.path.join(output, f'S{subject}-ADL{run}-AUGMENTED.csv'),
        index=False,
        header=False
    )


def __write_meta(df: pd.DataFrame, output: Text, subject: int, run: int) -> None:
    df.to_csv(
        os.path.join(output, f'S{subject}-ADL{run}-META.csv'),
        index=True,
        columns=[data.MID_LEVEL_COLUMN, 'ocd'],
        header=['activity', 'ocd']
    )


def main() -> None:
    args = __arg_parser().parse_args()

    # Outer
    outer_state = augmentation.Stateful(
        'outer',
        'Outer Action',
        None,
        None
    )

    # Clean Table
    clean_table_state_machine = augmentation.one_state_action_state_machine_fn(
        'clean_table',
        'Clean Table',
        outer_state
    )

    # Toggle Switch
    switch_state_machine = augmentation.one_state_action_state_machine_fn(
        'toggle_switch',
        'Toggle Switch',
        outer_state
    )

    # Door 1
    door_one_state_machine = augmentation.two_state_action_state_machine_fn(
        'door_1',
        'Open Door 1',
        'Close Coor 1',
        outer_state
    )

    # Dishwasher
    dishwasher_state_machine = augmentation.two_state_action_state_machine_fn(
        'dishwasher',
        'Open Dishwasher',
        'Close Dishwasher',
        outer_state
    )

    # Fridge
    fridge_state_machine = augmentation.two_state_action_state_machine_fn(
        'fridge',
        'Open Fridge',
        'Close Fridge',
        outer_state
    )

    # --- Subject 1 ---
    # toggle switch, clean table, open/close fridge
    # magic number: 4
    subject_one_state_machine = partial(
        augmentation.action_state_machine,
        state_machine_fn={
            'toggle_switch': (switch_state_machine, 4),
            'clean_table': (clean_table_state_machine, 4),
            'fridge': (fridge_state_machine, 4)
        },
        outer_state=outer_state
    )

    subject_one_collect_fn = partial(
        augmentation.collect_actions,
        state_machine_fn=subject_one_state_machine,
        outer_state=outer_state
    )

    # --- Subject 2 ---
    # repeat open/close door 1 (and NOT door 2)
    # further compulsions: open/close fridge, open/close dishwasher
    # magic number: 3
    subject_two_state_machine = partial(
        augmentation.action_state_machine,
        state_machine_fn={
            'door_1': (door_one_state_machine, 3),
            'fridge': (fridge_state_machine, 3),
            'dishwasher': (dishwasher_state_machine, 3)
        },
        outer_state=outer_state
    )

    subject_two_collect_fn = partial(
        augmentation.collect_actions,
        state_machine_fn=subject_two_state_machine,
        outer_state=outer_state
    )

    # --- Subject 3 ---
    # different magic numbers for compulsions
    # toggle switch: 5
    # clean table: 4
    # open / close dishwasher: 3
    # open / close fridge: 3
    subject_three_state_machine = partial(
        augmentation.action_state_machine,
        state_machine_fn={
            'toggle_switch': (switch_state_machine, 5),
            'clean_table': (clean_table_state_machine, 4),
            'dishwasher': (dishwasher_state_machine, 3),
            'fridge': (fridge_state_machine, 3)
        },
        outer_state=outer_state
    )

    subject_three_collect_fn = partial(
        augmentation.collect_actions,
        state_machine_fn=subject_three_state_machine,
        outer_state=outer_state
    )

    # Subject 4 - healthy, no OCD behavior

    collect_fns = [
        subject_one_collect_fn,
        subject_two_collect_fn,
        subject_three_collect_fn,
        None
    ]

    file_regex = re.compile(f'S(\d)-ADL(\d).dat')
    
    for subject, collect_fn in enumerate(collect_fns, start=1):
        ids, adls = __get_files(file_regex, os.path.join(args.path, f'S{subject}-ADL?.dat'))

        if collect_fn:
            drill = augmentation.read_dat(os.path.join(args.path, f'S{subject}-Drill.dat'))
            augmented = augmentation.augment(adls, drill, collect_fn, args.include_original)

        else:
            augmented = [adl.assign(ocd=0) for adl in adls]

        for run, df in zip(ids, augmented):
            __write_augmented(df, args.output, subject, run)
            __write_meta(df, args.output, subject, run)


if __name__ == "__main__":
    main()
