from collections import defaultdict, namedtuple
import csv
from itertools import cycle
from functools import partial, reduce
from typing import Callable, Dict, List, Set, Text, Tuple

import matplotlib.pylab as plt
import numpy as np
import pandas as pd
import tensorflow as tf

from ocddetection.data import preprocessing


Stateful = namedtuple('State', ['group', 'action', 'start', 'end'])
Action = namedtuple('Action', ['start', 'end'])

TABLE = dict(preprocessing.MID_LEVEL_LABELS)

TRANSITION_FN = Callable[[pd.Timedelta, pd.Timedelta], Stateful]

def __map_fn(t):
    return tf.gather(t, [0] + preprocessing.SENSORS + [preprocessing.MID_LEVEL])


def __filter_fn(t):
    return not tf.math.reduce_any(tf.math.is_nan(t))


def __dataset(path: Text) -> tf.data.Dataset:
    return preprocessing.to_dataset([path]).map(__map_fn).filter(__filter_fn)


def __dataframe(path: Text) -> pd.DataFrame:
    df = pd.DataFrame(__dataset(path).as_numpy_iterator())
    df[df.columns[-1]] = df[df.columns[-1]].astype('category')
    df[df.columns[-1]].cat.categories = [TABLE[int(cat)] for cat in df[df.columns[-1]].cat.categories]
    
    return df.set_index(pd.to_timedelta(df[0], 'ms')).drop(columns=[0])


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


def __action_state_machine(
    state: Stateful,
    index: pd.Timedelta,
    item: Text,
    actions: Dict[Text, List[Action]],
    state_machine_fn: Dict[Text, Callable[[Stateful, pd.Timedelta, Text, List[Action]], Tuple[Stateful, List[Action]]]],
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


def __collect_actions(
    df: pd.Series,
    state_machine_fn: Callable[[Stateful, pd.Timedelta, Text, List[Action]], Tuple[Stateful, Dict[Text, List[Action]]]],
    outer_state: Stateful
) -> Dict[Text, List[Action]]:
    return reduce(
        lambda s, a: state_machine_fn(s[0], a[0], a[1], s[1]),
        df[df.columns[-1]].items(),
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


def __merge_actions(adl, adl_actions, drill, drill_actions, num_repetitions: int = 3):
    def __reduce_fn(s, a):
        activity = adl \
            .loc[s[0]:a[1].end] \
            .assign(ocd=0)
        
        repetitions = __repeat_actions(
            [next(drill_actions[a[0]]) for _ in range(num_repetitions)],
            drill,
            a[1].end
        ).assign(ocd=1)
        
        window = __reindex_by_offset(pd.concat([activity, repetitions]), s[1])

        return (a[1].end + pd.Timedelta('0.01s'), window.index[-1] + pd.Timedelta('0.01s'), s[2] + [window])

    adl_end, repetition_end, windows = reduce(__reduce_fn, adl_actions, (pd.Timedelta('-0.03s'), pd.Timedelta('-0.03s'), []))
    windows.append(__reindex_by_offset(adl.loc[adl_end:].assign(ocd=0), repetition_end))

    return pd.concat(windows)


def augment(
    adls: List[pd.DataFrame],
    drill: pd.DataFrame,
    action_collection_fn: Callable[[pd.DataFrame], Dict[Text, List[Action]]],
    num_repetitions: int = 3
):
    adls_actions =[
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
                num_repetitions
            ),
            zip(adls, adls_actions)
        )
    )
