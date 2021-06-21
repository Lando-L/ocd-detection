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

    parser.add_argument('--num-repetitions', type=int, default=3)
    parser.add_argument('--include-original', dest='include_original', action='store_const', const=True, default=True)
    parser.add_argument('--exclude-original', dest='include_original', action='store_const', const=False, default=True)

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
    clean_table_state_machine = augmentation.one_state_action_state_machine_fn(
        'clean_table',
        'Clean Table',
        outer_state
    )

    # Toggle Switch
    fridge_state_machine = augmentation.one_state_action_state_machine_fn(
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
            augmented = augmentation.augment(adls, drill, collect_fn, args.num_repetitions, args.include_original)

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
