import gc

import pandas as pd
import numpy as np
from typing import Tuple


class MissingValueGenerator:

    def get_missing(self, df: np.ndarray,
                    flat_miss_mask: np.array) -> Tuple[pd.DataFrame, np.ndarray]:
        """
        Get missing data for original data set.

        Return df_with nan and miss_mask
        """
        miss_flat = np.copy(df)
        miss_flat = miss_flat.reshape(-1, )
        miss_flat[flat_miss_mask] = np.nan
        cols = ["_".join(['x', str(i)]) for i in range(df.shape[1])]
        df_miss = pd.DataFrame(data=miss_flat.reshape(df.shape), columns=cols)
        del df, miss_flat
        gc.collect()
        return df_miss, flat_miss_mask

    def get_flat_miss_mask(self, df: np.ndarray,
                           p_missing: float = None) -> np.array:
        """
        Param p_missing - missing proportion.

        If it equal 0.2 It means that 20% dataset values are missed.

        """
        df_size = df.size
        miss_size = int(df_size * p_missing)
        indx = np.arange(df_size)
        flat_miss_mask = np.random.choice(indx, miss_size, replace=False)
        return flat_miss_mask

    def run(self, df_origin: np.ndarray, p_missing: float = None) -> Tuple[pd.DataFrame, np.array]:
        """
        Run whole missing generator

        Return df_with nan and miss_mask
        """
        mask_miss = self.get_flat_miss_mask(df_origin, p_missing)
        return self.get_missing(df_origin, mask_miss)
