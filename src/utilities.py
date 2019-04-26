import logging
import sys

from scipy import stats
import pandas as pd
import numpy as np

import constants
from terminator import MissingValueGenerator


def mode(arr):
    return stats.mode(arr)[0][0]


def constant(arr):
    return constants.CONSTANT


def setup_logger():
    """Sets up the logger handlers for jupyter notebook, ipython or python.
       The logging level is set to INFO."""
    logger = logging.getLogger()
    logger.handlers = []
    streamhandler = logging.StreamHandler(sys.stdout)
    streamhandler.setFormatter(logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%m/%d/%Y %I:%M:%S %p'))
    streamhandler.setLevel(logging.INFO)
    logger.addHandler(streamhandler)
    logger.setLevel(logging.INFO)
    # Set requests and urllib logs to WARN level to not pollute logs
    logging.getLogger("requests").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("botocore").setLevel(logging.WARNING)
    logging.getLogger("boto3").setLevel(logging.WARNING)


def intervals(df_in, inplace=False) -> pd.DataFrame:
    if not inplace:
        data = df_in.copy()
    else:
        data = df_in
    for col in data.columns:
        std_col = np.std(data[col])
        bins = 1
        if std_col != 0:
            bins = int(1 / std_col)
        if bins > 20:
            bins = 20
        bins_names = ['_'.join([col, str(x_bin)]) for x_bin in range(bins)]
        data[col] = pd.cut(data[col], bins=bins, labels=bins_names)
    return data


def concat_columns(df, sep=' ') -> pd.DataFrame:
    res = df.apply(lambda x: sep.join(x.dropna()), axis=1)
    res = res.str.split()
    return res


def stochastic_corruption(ndarr) -> np.ndarray:
    # 0.5 set in article
    x: pd.DataFrame = MissingValueGenerator().run(ndarr, 0.3)[0]
    # get constant
    cor_ndarr = x.fillna(0)
    return cor_ndarr.values
