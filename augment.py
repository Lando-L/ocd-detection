from argparse import ArgumentParser
from functools import partial
import os

import tensorflow as tf

from ocddetection import data
from ocddetection.data import augmentation


def __arg_parser() -> ArgumentParser:
    parser = ArgumentParser()
    
    parser.add_argument('path', type=str)
    parser.add_argument('output', type=str)

    parser.add_argument('--repetition-rate', type=float, default=.75)
    parser.add_argument('--num-repetitions', type=int, default=3)

    return parser


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
    clean_table_state_fn = augmentation.one_state_action_fn(
        'clean_table',
        'Clean Table'
    )

    clean_table_state_machine = partial(
        augmentation.one_state_action_state_machine,
        state_action='Clean Table',
        state_fn=clean_table_state_fn,
        outer_state=outer_state
    )

    # Drawers
    drawer_one_open_state_fn, drawer_one_null_state_fn, drawer_one_close_state_fn = augmentation.two_state_action_fn(
        'drawer_one',
        'Open Drawer 1',
        'Close Drawer 1'
    )

    drawer_one_state_machine = partial(
        augmentation.two_state_action_state_machine,
        open_action='Open Drawer 1',
        open_state_fn=drawer_one_open_state_fn,
        padding_action='Null',
        padding_state_fn=drawer_one_null_state_fn,
        close_action='Close Drawer 1',
        close_state_fn=drawer_one_close_state_fn,
        outer_state=outer_state
    )

    drawer_two_open_state_fn, drawer_two_null_state_fn, drawer_two_close_state_fn = augmentation.two_state_action_fn(
        'drawer_two',
        'Open Drawer 2',
        'Close Drawer 2'
    )

    drawer_two_state_machine = partial(
        augmentation.two_state_action_state_machine,
        open_action='Open Drawer 2',
        open_state_fn=drawer_two_open_state_fn,
        padding_action='Null',
        padding_state_fn=drawer_two_null_state_fn,
        close_action='Close Drawer 2',
        close_state_fn=drawer_two_close_state_fn,
        outer_state=outer_state
    )

    drawer_three_open_state_fn, drawer_three_null_state_fn, drawer_three_close_state_fn = augmentation.two_state_action_fn(
        'drawer_three',
        'Open Drawer 3',
        'Close Drawer 3'
    )

    drawer_three_state_machine = partial(
        augmentation.two_state_action_state_machine,
        open_action='Open Drawer 3',
        open_state_fn=drawer_three_open_state_fn,
        padding_action='Null',
        padding_state_fn=drawer_three_null_state_fn,
        close_action='Close Drawer 3',
        close_state_fn=drawer_three_close_state_fn,
        outer_state=outer_state
    )

    # Dishwasher
    dishwasher_open_state_fn, dishwasher_null_state_fn, dishwasher_close_state_fn = augmentation.two_state_action_fn(
        'dishwasher',
        'Open Dishwasher',
        'Close Dishwasher'
    )

    dishwasher_state_machine = partial(
        augmentation.two_state_action_state_machine,
        open_action='Open Dishwasher',
        open_state_fn=dishwasher_open_state_fn,
        padding_action='Null',
        padding_state_fn=dishwasher_null_state_fn,
        close_action='Close Dishwasher',
        close_state_fn=dishwasher_close_state_fn,
        outer_state=outer_state
    )

    # Fridge
    fridge_open_state_fn, fridge_null_state_fn, fridge_close_state_fn = augmentation.two_state_action_fn(
        'fridge',
        'Open Fridge',
        'Close Fridge'
    )

    fridge_state_machine = partial(
        augmentation.two_state_action_state_machine,
        open_action='Open Fridge',
        open_state_fn=fridge_open_state_fn,
        padding_action='Null',
        padding_state_fn=fridge_null_state_fn,
        close_action='Close Fridge',
        close_state_fn=fridge_close_state_fn,
        outer_state=outer_state
    )

    # Subject 1
    subject_one_state_machine = partial(
        augmentation.action_state_machine,
        state_machine_fn={
            'drawer_one': drawer_one_state_machine,
            'drawer_two': drawer_two_state_machine,
            'drawer_three': drawer_three_state_machine
        },
        outer_state=outer_state
    )

    subject_one_collect_fn = partial(
        augmentation.collect_actions,
        state_machine_fn=subject_one_state_machine,
        outer_state=outer_state
    )

    # Subject 2
    subject_two_state_machine = partial(
        augmentation.action_state_machine,
        state_machine_fn={
            'clean_table': clean_table_state_machine,
            'fridge': fridge_state_machine
        },
        outer_state=outer_state
    )

    subject_two_collect_fn = partial(
        augmentation.collect_actions,
        state_machine_fn=subject_two_state_machine,
        outer_state=outer_state
    )

    # Subject 3
    subject_three_state_machine = partial(
        augmentation.action_state_machine,
        state_machine_fn={
            'dishwasher': dishwasher_state_machine,
            'fridge': fridge_state_machine
        },
        outer_state=outer_state
    )

    subject_three_collect_fn = partial(
        augmentation.collect_actions,
        state_machine_fn=subject_three_state_machine,
        outer_state=outer_state
    )

    collect_fns = [
        subject_one_collect_fn,
        subject_two_collect_fn,
        subject_three_collect_fn,
        None
    ]
    
    for subject, collect_fn in enumerate(collect_fns, start=1):
        adls = list(map(augmentation.read_dat, tf.io.gfile.glob(os.path.join(args.path, f'S{subject}-ADL?.dat'))))

        if collect_fn:
            drill = augmentation.read_dat(os.path.join(args.path, f'S{subject}-Drill.dat'))
            augmented = augmentation.augment(adls, drill, collect_fn, args.repetition_rate, args.num_repetitions)

        else:
            augmented = [adl.assign(ocd=0) for adl in adls]

        for run, df in enumerate(augmented, start=1):
            df.drop(data.MID_LEVEL_COLUMN, axis=1).to_csv(
                os.path.join(args.output, f'S{subject}-ADL{run}-AUGMENTED.csv'),
                index=False,
                header=False
            )

            df.to_csv(
                os.path.join(args.output, f'S{subject}-ADL{run}-META.csv'),
                index=True,
                columns=[data.MID_LEVEL_COLUMN, 'ocd'],
                header=['activity', 'ocd']
            )


if __name__ == "__main__":
    main()
