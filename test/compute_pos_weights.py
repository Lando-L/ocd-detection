import pandas as pd

total_counts = {0: 0,
                1: 0}
for s in range(1, 5):
    class_counts = {0: 0,
                    1: 0}
    for r in range(1, 6):
        ds_aug = pd.read_csv(f'../../dsets/opportunity/augmented/icmla/S{s}-ADL{r}-META.csv')
        ds_counts = ds_aug['ocd'].value_counts()
        class_counts[0] += ds_counts.iloc[0]
        total_counts[0] += ds_counts.iloc[0]
        if s != 3:
            class_counts[1] += ds_counts.iloc[1]
            total_counts[1] += ds_counts.iloc[1]
        else:
            class_counts[1] = 1
    print(f'Subject {s}: pos_weight: {class_counts[0] / class_counts[1]:.2f}')
print(f'Total pos_weight: {total_counts[0] / total_counts[1]:.2f}')
