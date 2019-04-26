import logging
from typing import List

import numpy as np
import pandas as pd
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import MinMaxScaler

import preparation
from imputation_configs import get_model
from evaluation import Evaluator
from terminator import MissingValueGenerator
import matplotlib.pyplot as plt
import seaborn as sns


class ExperimentPipeline:
    SCORE_LIST = ['rmse', 'sce', 'uce']

    def __init__(self, imputer_abbr: str = None, **imputer_params):
        self.strategy_abbr = imputer_abbr
        try:
            self.model_factory = get_model(self.strategy_abbr, **imputer_params)
        except AttributeError:
            self.logger.info(f"{imputer_abbr} not in expected list")
        self.logger = logging.getLogger(self.__class__.__name__)

    def run_grid_search_plot(self,
                             data: pd.DataFrame,
                             miss_prop_range,
                             target_col_name: str = None,
                             imputer_abbr: List[str] = None,
                             verbose=False,
                             **imputer_params):
        """
        Start evaluation for all models from imputer_abbr.

        :param data:
        :param miss_prop_range:
        :param target_col_name:
        :param imputer_abbr:
        :param verbose:
        :param imputer_params:
        :return:
        """
        result_df = pd.DataFrame()
        for abbr in imputer_abbr:
            print(abbr)
            self.strategy_abbr = abbr
            self.model_factory = get_model(self.strategy_abbr, **imputer_params)
            for prop in miss_prop_range:
                result = self.run(data,
                                  target_col=target_col_name,
                                  miss_proportion=prop,
                                  verbose=verbose)
                result_df = result_df.append(pd.DataFrame([result]), ignore_index=True)
        return result_df

    def run(self, df_original,
            target_col='target',
            miss_proportion=None,
            verbose=False):

        imputation_res = self.get_imputation_results(df_original=df_original,
                                                     target_col=target_col,
                                                     miss_proportion=miss_proportion)
        if verbose:
            self.logger.info(f'\n {self.strategy_abbr} strategy, \n With missing proportion: {miss_proportion}')

        result = self.get_evaluation_metrics(**imputation_res,
                                             verbose=verbose,
                                             m_prop=miss_proportion)
        return result

    def get_imputation_results(self,
                               df_original,
                               target_col='target',
                               miss_proportion: float = None):
        """
        Take original dataframe and lounch pipeline to
        stages = [MissingValueGenerator, Imputation, Evaluation]

        :param df_original:
        :param target_col:
        :param miss_proportion:
        :return: scores for three diff metrics of imputation quality
        """

        features_or, target = preparation.data_preparation(df_original, target_col)
        columns = list(features_or)
        # todo: train test split
        indices = np.arange(features_or.shape[0]).reshape(-1, 1)
        train, test = train_test_split(indices, shuffle=False, test_size=0.3, random_state=12)
        train_fts_ = features_or.iloc[np.ravel(train), :]
        test_fts_ = features_or.iloc[np.ravel(test), :]
        y_train = target[np.ravel(train)]
        y_test = target[np.ravel(test)]
        scaler = MinMaxScaler()
        train_fts = scaler.fit_transform(train_fts_)
        test_fts = scaler.transform(test_fts_)
        m_df_train, mask_missing_train = MissingValueGenerator().run(train_fts, miss_proportion)
        m_df_test, mask_missing_test = MissingValueGenerator().run(test_fts, miss_proportion)
        # todo: add preprocessing for categorical data
        # ohe
        # fit models on train impute on test stage
        imputer = self.model_factory()
        features_imputed = imputer.complete(m_df_test)
        if self.strategy_abbr != 'emb':
            features_imputed.columns = columns
        self.logger.info(f'{self.strategy_abbr.upper()} --- Starting evaluation...')
        result = dict(df_original=test_fts,
                      df_imputed=features_imputed,
                      target=y_test,
                      mask_missing=mask_missing_test)
        return result

    # todo: add verbose param
    def get_evaluation_metrics(self, df_original, df_imputed, target, mask_missing, m_prop, verbose):
        """
        Generate evaluation metrics for datasets

        :param m_prop:
        :param verbose:
        :param target:
        :param df_original:
        :param df_imputed:
        :param mask_missing:
        :return:
        """
        results = dict()
        results['prop'] = m_prop
        results['strategy'] = self.strategy_abbr
        # todo: refactor it with score factory
        if self.strategy_abbr not in ['constant', 'emb']:
            results['rmse'] = Evaluator().get_compare_metrics(df_original, df_imputed, mask_missing)
        if self.strategy_abbr not in ['emb']:
            results['uce'] = Evaluator().uce(df_original, df_imputed)
            results['silhouette'] = Evaluator().silhouette(df_imputed)

        # todo: add pipeline for regression with auto detect the target type
        sce_or = Evaluator().sce(df_original, target)
        sce_im = Evaluator().sce(df_imputed, target)
        results['sce'] = sce_im - sce_or
        results['f1'] = Evaluator().f1_score(df_imputed, target)
        # if verbose:
        #     self.logger.info(f'UCE - clustering error between original and imputed datasets = ', np.round(results['uce'], 5))
        #     self.logger.info(f'RMSE score between original values and imputed = ', np.round(results['rmse'], 5))
        #     self.logger.info(f'SCE - classification error between original and imputed datasets', np.round(results['sce'], 5))
        return results

    def viz_lineplot_error(self, df: pd.DataFrame):
        """

        :param df: result df
        :return:
        """
        if df is not None:
            n_plot_rows = len(self.SCORE_LIST)
            f, axes = plt.subplots(n_plot_rows, 1, figsize=(12, 18))
            f.subplots_adjust(hspace=0.3, wspace=0.1)
            sns.set_style("darkgrid")
            sns.set_context(font_scale=1.5)
            for j in range(n_plot_rows):
                score = self.SCORE_LIST[j]
                sns.lineplot('prop', score, hue='strategy', style="strategy",
                             markers=True, dashes=False, data=df[~pd.isna(df[score])], ax=axes[j])
                axes[j].legend(loc="upper left")
                axes[j].set_title(f'Metric: {score.upper()}')
                axes[j].set(xlabel='Missing data proportion', ylabel=f'{score.upper()}')

    def viz_model_competition(self, df):
        if df is not None:
            scores = ['f1', 'silhouette']
            n_plot_rows = len(scores)
            f, axes = plt.subplots(n_plot_rows, 1, figsize=(12, 14))
            f.subplots_adjust(hspace=0.3, wspace=0.1)
            sns.set_style("darkgrid")
            sns.set_context(font_scale=1.5)
            for j in range(n_plot_rows):
                score = scores[j]
                sns.lineplot('prop', score, hue='strategy', style="strategy",
                             markers=True, dashes=False, data=df[~pd.isna(df[score])], ax=axes[j])
                axes[j].legend(loc="upper left")
                axes[j].set_title(f'Metric: {score.upper()}')
                axes[j].set(xlabel='Missing data proportion', ylabel=f'{score.upper()}')
