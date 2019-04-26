from typing import List

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler


def data_preparation(df, target_col_name):
    features: pd.DataFrame = df.drop([target_col_name], axis=1)
    columns: List[str] = ['_'.join(['x', str(x)]) for x in features.columns]
    features: np.ndarray = MinMaxScaler().fit_transform(features)
    features: pd.DataFrame = pd.DataFrame(features, columns=columns)
    features = features.reset_index(drop=True)
    target = df[target_col_name]
    if target.dtype == object:
        target = LabelEncoder().fit_transform(target)
    return features, target
