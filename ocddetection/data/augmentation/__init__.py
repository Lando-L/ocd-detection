from collections import namedtuple
import csv
from itertools import cycle
from functools import partial, reduce
from typing import Callable, List, Set, Text

import matplotlib.pylab as plt
import numpy as np
import pandas as pd
import tensorflow as tf

from ocddetection.data import preprocessing


TABLE = dict(preprocessing.MID_LEVEL_LABELS)
Stateful = namedtuple('State', ['action', 'start', 'end'])
Action = namedtuple('Action', ['start', 'end'])

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


def __one_state_action(action: Text):
    def state(start: pd.Timedelta, end: pd.Timedelta) -> Stateful:
        return Stateful(action, start, end)

    return state


def __two_state_action(action_open: Text, action_close: Text):
    def open_state(start: pd.Timedelta, end: pd.Timedelta) -> Stateful:
        return Stateful(action_open, start, end)

    def null_state(start: pd.Timedelta, end: pd.Timedelta) -> Stateful:
        return Stateful('Null', start, end)

    def close_state(start: pd.Timedelta, end: pd.Timedelta) -> Stateful:
        return Stateful(action_close, start, end)
    
    return open_state, null_state, close_state


def __action_selection(actions: Set[Text], df: pd.DataFrame) -> pd.Series:
    return df[df[df.columns[-1]].isin(actions)][df.columns[-1]]


def __one_state_action_windows(
    ss: pd.Series,
    action: Text,
    one_state_action_fn: Callable[[pd.Timedelta, pd.Timedelta], Stateful]
) -> List[Action]:
    state = one_state_action_fn(None, None)
    actions = []

    for index, value in ss.items():
        if value == action:
            


def __action_windows(ss: pd.Series) -> List[Action]:
    state = __open_state(None, None)
    actions = []

    for index, value in ss.items():
        if value == 'Open Fridge':
            if state.action == 'Null':
                state = __open_state(index, index)

            elif state.action == 'Close Fridge':
                actions.append(Action(state.start, state.end))
                state = __open_state(index, index)
            
            else:
                state = __open_state(state.start or index, index)
                    
        elif value == 'Null':
            if state.action == 'Close Fridge':
                actions.append(Action(state.start, state.end))
                state = __null_state(None, None)
                
            else:
                state = __null_state(state.start, index)
            
        elif value == 'Close Fridge':
            state = __close_state(state.start, index)
    
    return actions


def __actions(df: pd.DataFrame) -> List[Action]:
    return __action_windows(__action_selection(df))


def __reindex_by_offset(df: pd.DataFrame, offset: pd.Timedelta):
    df.index += offset + pd.Timedelta('0.03s') - df.index[0]

    return df


def __repeated_actions(drill_actions: List[Action], drill: pd.DataFrame, offset: pd.Timedelta) -> pd.DataFrame:
    def __reduce_fn(t, action):
        window = __reindex_by_offset(drill.loc[action.start:action.end], t[0])
        return (window.index[-1], t[1] + [window])
    
    return pd.concat(reduce(__reduce_fn, drill_actions, (offset, []))[1])


def __merge_actions(adl, adl_actions, drill, drill_actions, num_repetitions: int = 3):
    def __reduce_fn(state, action):
        activity = adl.loc[state[0]:action.end].assign(ocd=0)
        repetitions = __repeated_actions([next(drill_actions) for _ in range(num_repetitions)], drill, action.end).assign(ocd=1)
        window = __reindex_by_offset(pd.concat([activity, repetitions]), state[1])
        return (action.end + pd.Timedelta('0.01s'), window.index[-1] + pd.Timedelta('0.01s'), state[2] + [window])

    adl_end, repetition_end, windows = reduce(__reduce_fn, adl_actions, (pd.Timedelta('-0.03s'), pd.Timedelta('-0.03s'), []))
    windows.append(__reindex_by_offset(adl.loc[adl_end:].assign(ocd=0), repetition_end))

    return pd.concat(windows)


def __augment(adls: List[pd.DataFrame], drill: pd.DataFrame, num_repetitions: int = 3):
    adls_action = list(map(__actions, adls))
    drill_action = __actions(drill)

    assert all(len(x) * num_repetitions < len(drill_action) for x in adls_action)

    drill_action_cycle = cycle(drill_action)

    return list(
        map(
            lambda x: __merge_actions(
                x[0],
                x[1],
                drill,
                iter(next(drill_action_cycle) for _ in range(len(x[1]) * num_repetitions)),
                num_repetitions
            ),
            zip(adls, adls_action)
        )
    )
