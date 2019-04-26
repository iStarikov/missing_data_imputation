"""
Main entry to evaluate all pipelines jn all data
"""
import argparse
from typing import List

import numpy as np
import pandas as pd

import utilities
from experiments import ExperimentPipeline
from imputation_configs import get_model


def parse_args() -> argparse.Namespace:
    """
    Parses command line arguments
    :return: Namespace
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--strategies", nargs='?', type=List[str], default=['median', 'constant', 'knn', 'svd', 'mf', 'mice'],
        help="Set the default strategies for imputation"
    )
    parser.add_argument(
        "--missing_proportion",
        type=List[float],
        default=[np.linspace(0.2, 0.9, 8)],
        help="Lost of missingness proportion"
    )
    return parser.parse_args()


def run_grid_search_plot(data: pd.DataFrame,
                         target_col_name: str = None,
                         miss_prop_range: List[float] = None,
                         imputer_abbr: List[str] = None,
                         verbose=False,
                         **imputer_params):
    clargs = parse_args()

    experiment_pipeline = ExperimentPipeline()
    experiment_pipeline.result_df = pd.DataFrame()
    data = data.copy()
    for abbr in imputer_abbr:
        print(abbr)
        experiment_pipeline.strategy_abbr = abbr
        experiment_pipeline.model_factory = get_model(experiment_pipeline.strategy_abbr, **imputer_params)
        for prop in miss_prop_range:
            result = experiment_pipeline.run(data,
                                             target_col=target_col_name,
                                             miss_proportion=prop,
                                             verbose=verbose)
            experiment_pipeline.result_df = experiment_pipeline.result_df.append(pd.DataFrame([result]),
                                                                                 ignore_index=True)

    experiment_pipeline.viz_lineplot_error(experiment_pipeline.result_df)
    return experiment_pipeline.result_df


def main() -> None:
    """
    Whole pipeline start and end
    :return: None
    """
    datas = None
    target_col_names = None
    miss_prop_range = np.linspace(0.1, 0.9, 9)
    imputer_abbr = ['constant', 'median', 'knn', 'svd', 'mf', 'mice']
    imputer_params = {}
    result = pd.DataFrame()
    for data, target in zip(datas, target_col_names):
        res = run_grid_search_plot(data, target, miss_prop_range,
                                   imputer_abbr, verbose=False, ** imputer_params)
    result = result.append(res, ignore_index=True)


if __name__ == '__main__':
    utilities.setup_logger()
    main()
