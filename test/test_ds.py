import pandas as pd
import sys

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

ocd_act = {
    4: 'toggle switch, clean table, open/close fridge',
    1: 'open/close door 1, open/close fridge, open/close dishwasher',
    2: 'toggle switch, open/close dishwasher, open/close fridge',
    3: 'none'
}

def read_dat(path) -> pd.DataFrame:
    df = pd.read_csv(path, sep=' ', index_col=0, header=None, usecols=[0] + SENSORS + [MID_LEVEL_COLUMN]).dropna()
    df = df.set_index(pd.to_timedelta(df.index, 'ms'))
    df[MID_LEVEL_COLUMN] = df[MID_LEVEL_COLUMN].astype('category')
    df[MID_LEVEL_COLUMN].cat.categories = [MID_LEVEL_LABELS[int(cat)] for cat in df[MID_LEVEL_COLUMN].cat.categories]

    return df

for subject in range(1, 5):
    print(f'Subject {subject} OCD activities: {ocd_act[subject]}')
    for run in range(1, 6):
        print(f'S{subject}-ADL{run}\\')
        ds_aug = pd.read_csv(f'../{sys.argv[1]}/S{subject}-ADL{run}-META.csv')
        ds_orig = read_dat(f'../../../dsets/opportunity/dataset/S{subject}-ADL{run}.dat')

        diff_df = ds_aug['activity'].value_counts() - ds_orig[MID_LEVEL_COLUMN].value_counts()
        print(diff_df[diff_df != 0])
        print('\\')
