from collections import defaultdict, namedtuple
from itertools import cycle
from functools import partial, reduce
from typing import Callable, Dict, List, Text, Tuple

import pandas as pd

from ocddetection.data import MID_LEVEL_COLUMN, MID_LEVEL_LABELS, SENSORS


Stateful = namedtuple('State', ['group', 'action', 'start', 'end'])
Action = namedtuple('Action', ['start', 'end'])

TRANSITION_FN = Callable[[pd.Timedelta, pd.Timedelta], Stateful]
STATE_MACHINE_FN = Callable[[Stateful, pd.Timedelta, Text, List[Action]], Tuple[Stateful, List[Action]]]


def read_dat(path: Text) -> pd.DataFrame:
    df = pd.read_csv(path, sep=' ', index_col=0, header=None, usecols=[0] + SENSORS + [MID_LEVEL_COLUMN]).dropna()
    df = df.set_index(pd.to_timedelta(df.index, 'ms'))
    df[MID_LEVEL_COLUMN] = df[MID_LEVEL_COLUMN].astype('category')
    df[MID_LEVEL_COLUMN].cat.categories = [MID_LEVEL_LABELS[int(cat)] for cat in df[MID_LEVEL_COLUMN].cat.categories]
    
    return df


def __one_state_action_fn(group: Text, action: Text) -> TRANSITION_FN:
    def state(start: pd.Timedelta, end: pd.Timedelta) -> Stateful:
        return Stateful(group, action, start, end)

    return state


def __two_state_action_fn(
    group: Text,
    action_open: Text,
    action_close: Text
) -> Tuple[TRANSITION_FN, TRANSITION_FN, TRANSITION_FN]:
    def open_state(start: pd.Timedelta, end: pd.Timedelta) -> Stateful:
        return Stateful(group, action_open, start, end)

    def null_state(start: pd.Timedelta, end: pd.Timedelta) -> Stateful:
        return Stateful(group, 'Null', start, end)

    def close_state(start: pd.Timedelta, end: pd.Timedelta) -> Stateful:
        return Stateful(group, action_close, start, end)
    
    return open_state, null_state, close_state


def __one_state_action_state_machine(
    state: Stateful,
    index: pd.Timedelta,
    item: Text,
    actions: List[Action],
    state_action: Text,
    state_fn: TRANSITION_FN,
    outer_state: Stateful
) -> Tuple[Stateful, List[Action]]:
    if state.action == state_action:
        if item == state_action:
            return state_fn(state.start, index), actions
        
        else:
            return outer_state, actions + [Action(state.start, state.end)]
    
    else:
        if item == state_action:
            return state_fn(index, index), actions
        
        else:
            return outer_state, actions


def __two_state_action_state_machine(
    state: Stateful,
    index: pd.Timedelta,
    item: Text,
    actions: List[Action],
    open_action: Text,
    open_state_fn: TRANSITION_FN,
    padding_action: Text,
    padding_state_fn: TRANSITION_FN,
    close_action: Text,
    close_state_fn: TRANSITION_FN,
    outer_state: Stateful
) -> Tuple[Stateful, List[Action]]:
    if state.action == open_action:
        if item == open_action:
            return open_state_fn(state.start, index), actions
        
        elif item == padding_action:
            return padding_state_fn(state.start, index), actions
        
        elif item == close_action:
            return close_state_fn(state.start, index), actions
        
        else:
            return outer_state, actions
    
    elif state.action == padding_action:
        if item == open_action:
            return open_state_fn(index, index), actions
        
        elif item == padding_action:
            return padding_state_fn(state.start, index), actions
        
        elif item == close_action:
            return close_state_fn(state.start, index), actions
        
        else:
            return outer_state, actions
        
    elif state.action == close_action:
        if item == open_action:
            return open_state_fn(index, index), actions + [Action(state.start, state.end)]
        
        elif item == close_action:
            return close_state_fn(state.start, index), actions
        
        else:
            return outer_state, actions + [Action(state.start, state.end)]
    
    else:
        if item == open_action:
            return open_state_fn(index, index), actions
        
        else:
            return outer_state, actions


def one_state_action_state_machine_fn(
    group: Text,
    action: Text,
    outer_state: Stateful
) -> STATE_MACHINE_FN:  
    return partial(
        __one_state_action_state_machine,
        state_action=action,
        state_fn=__one_state_action_fn(group, action),
        outer_state=outer_state
    )


def two_state_action_state_machine_fn(
    group: Text,
    action_open: Text,
    action_close: Text,
    outer_state: Stateful
) -> STATE_MACHINE_FN:
    one_open_state_fn, null_state_fn, close_state_fn = __two_state_action_fn(group, action_open, action_close)

    return partial(
        __two_state_action_state_machine,
        open_action=action_open,
        open_state_fn=one_open_state_fn,
        padding_action='Null',
        padding_state_fn=null_state_fn,
        close_action=action_close,
        close_state_fn=close_state_fn,
        outer_state=outer_state
    )


def action_state_machine(
    state: Stateful,
    index: pd.Timedelta,
    item: Text,
    actions: Dict[Text, List[Action]],
    state_machine_fn: Dict[Text, STATE_MACHINE_FN],
    outer_state: Stateful
) -> Tuple[Stateful, Dict[Text, List[Action]]]:
    def evaluate_alternative_states():
        for group, fn in state_machine_fn.items():
            alternative_state, _ = fn(state, index, item, actions[group])

            if alternative_state.group != outer_state.group:
                return alternative_state
        
        return outer_state

    next_state, next_actions = None, actions

    if state.group in state_machine_fn:
        next_state, next_actions[state.group] = state_machine_fn[state.group](state, index, item, actions[state.group])
    
    if (next_state is None) or (next_state.group == outer_state.group):
        next_state = evaluate_alternative_states()
    
    return next_state, next_actions


def collect_actions(
    df: pd.Series,
    state_machine_fn: Callable[[Stateful, pd.Timedelta, Text, List[Action]], Tuple[Stateful, Dict[Text, List[Action]]]],
    outer_state: Stateful
) -> Dict[Text, List[Action]]:
    return reduce(
        lambda s, a: state_machine_fn(s[0], a[0], a[1], s[1]),
        df[MID_LEVEL_COLUMN].items(),
        (outer_state, defaultdict(list))
    )[1]


def __reindex_by_offset(df: pd.DataFrame, offset: pd.Timedelta):
    df.index += offset + pd.Timedelta('0.03s') - df.index[0]

    return df


def __repeat_actions(drill_actions: List[Action], drill: pd.DataFrame, offset: pd.Timedelta) -> pd.DataFrame:
    def __reduce_fn(s, a):
        window = __reindex_by_offset(drill.loc[a.start:a.end], s[0])
        return (window.index[-1], s[1] + [window])
    
    return pd.concat(reduce(__reduce_fn, drill_actions, (offset, []))[1])


def __merge_actions(
    adl: pd.DataFrame,
    adl_actions: Dict[Text, List[Action]],
    drill: pd.DataFrame,
    drill_actions: Dict[Text, List[Action]],
    num_repetitions: int,
    include_original: bool
):
    def __reduce_fn(s, a):
        non_ocd = adl \
            .loc[s[0]:a[1].start][:-1] \
            .assign(ocd=0)
        
        original_activity = adl \
            .loc[a[1].start:a[1].end] \
            .assign(ocd=(1 if include_original else 0))
        
        repeated_activities = __repeat_actions(
            [next(drill_actions[a[0]]) for _ in range(num_repetitions)],
            drill,
            a[1].end
        ).assign(ocd=1)
        
        window = __reindex_by_offset(pd.concat([non_ocd, original_activity, repeated_activities]), s[1])

        return (a[1].end + pd.Timedelta('0.01s'), window.index[-1] + pd.Timedelta('0.01s'), s[2] + [window])

    adl_end, repetition_end, windows = reduce(__reduce_fn, adl_actions, (pd.Timedelta('-0.03s'), pd.Timedelta('-0.03s'), []))
    windows.append(__reindex_by_offset(adl.loc[adl_end:].assign(ocd=0), repetition_end))

    return pd.concat(windows)


def augment(
    adls: List[pd.DataFrame],
    drill: pd.DataFrame,
    action_collection_fn: Callable[[pd.DataFrame], Dict[Text, List[Action]]],
    num_repetitions: int,
    include_original: bool
):
    adls_actions = [
        sorted(
            [
                (group, action)
                for group, actions in action_collection_fn(adl).items()
                for action in actions
            ],
            key=lambda x: x[1].start
        )
        for adl in adls
    ]
    
    drill_actions = {
        group: cycle(actions)
        for group, actions in action_collection_fn(drill).items()
    }

    return list(
        map(
            lambda x: __merge_actions(
                x[0],
                x[1],
                drill,
                drill_actions,
                num_repetitions,
                include_original
            ),
            zip(adls, adls_actions)
        )
    )
