from argparse import ArgumentParser
from functools import partial
import os

from ocddetection import data
from ocddetection.data import augmentation


def __arg_parser() -> ArgumentParser:
    parser = ArgumentParser()
    
    parser.add_argument('path', type=str)
    parser.add_argument('output', type=str)

    parser.add_argument('--num-repetitions', type=int, default=4)

    return parser


def main() -> None:
    args = __arg_parser().parse_args()
    
    adl_files, drill_files = data.files(args.path)

    toggle_switch_state_fn = augmentation.one_state_action_fn(
        'toggle_switch',
        'Toggle Switch'
    )

    fridge_open_state_fn, fridge_null_state_fn, fridge_close_state_fn = augmentation.two_state_action_fn(
        'fridge',
        'Open Fridge',
        'Close Fridge'
    )

    dishwasher_open_state_fn, dishwasher_null_state_fn, dishwasher_close_state_fn = augmentation.two_state_action_fn(
        'dishwasher',
        'Open Dishwasher',
        'Close Dishwasher'
    )

    outer_state = augmentation.Stateful(
        'outer',
        'Outer Action',
        None,
        None
    )

    toggle_switch_state_machine = partial(
        augmentation.one_state_action_state_machine,
        state_action='Toggle Switch',
        state_fn=toggle_switch_state_fn,
        outer_state=outer_state
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

    state_machine = partial(
        augmentation.action_state_machine,
        state_machine_fn={
            'toggle_switch': toggle_switch_state_machine,
            'fridge': fridge_state_machine,
            'dishwasher': dishwasher_state_machine
        },
        outer_state=outer_state
    )

    collect_fn = partial(
        augmentation.collect_actions,
        state_machine_fn=state_machine,
        outer_state=outer_state
    )
    
    for subject in drill_files.index:
        drill = augmentation.to_dataframe(drill_files.loc[subject, 'path'])
        adls = [augmentation.to_dataframe(path) for path in adl_files.loc[subject, 'path'].values]
        augmented = augmentation.augment(adls, drill, collect_fn, args.num_repetitions)

        for run, df in zip(adl_files.loc[subject].index, augmented):
            df.drop(78, axis=1).to_csv(
                os.path.join(args.output, f'S{subject}-ADL{run}-AUGMENTED.csv'),
                index=False,
                header=False
            )

            df.to_csv(
                os.path.join(args.output, f'S{subject}-ADL{run}-META.csv'),
                index=True,
                columns=[78, 'ocd'],
                header=['activity', 'ocd']
            )


if __name__ == "__main__":
    main()
