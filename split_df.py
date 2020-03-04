import pandas as pd
from sklearn.model_selection import train_test_split


def split_df(df, test_size, random_state=10):
    train_data, test_data = train_test_split(
        df, test_size=test_size, random_state=random_state
    )
    val_data, test_data = train_test_split(
        test_data, test_size=0.5, random_state=random_state
    )
    for df in train_data, val_data, test_data:
        df.reset_index(drop=True, inplace=True)
    return train_data, val_data, test_data