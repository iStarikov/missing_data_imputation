import pandas as pd
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import normalized_mutual_info_score, roc_auc_score, mean_squared_error, silhouette_score, f1_score

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.preprocessing import OneHotEncoder


class Evaluator:
    def __init__(self, m_prop: float = None, method=None):
        self.m_prop = m_prop
        self.method: str = method

    def rmse(self, predictions, targets):
        # return np.mean(np.sqrt(((predictions - targets) ** 2).mean()))
        return np.sqrt(mean_squared_error(targets, predictions))

    def uce(self, df_origin, df_imputed):
        """
        Unsupervised clasterization error
        """
        clasters_or = AgglomerativeClustering().fit_predict(df_origin)
        clusters_imp = AgglomerativeClustering().fit_predict(df_imputed)
        uce = 1 - normalized_mutual_info_score(clasters_or, clusters_imp)
        return uce

    def sce(self, features, target):
        """
        Supervised classification error
        """
        clf = RandomForestClassifier().fit(features, target)
        # clf = LGBMClassifier().fit(feature, target)
        prediction = clf.predict_proba(features)
        if isinstance(target, pd.Series):
            target = target.values
        ohe = OneHotEncoder()
        target = ohe.fit_transform(target.reshape(-1, 1)).toarray()
        sce = 1 - roc_auc_score(target, prediction)
        return sce

    def get_compare_metrics(self, origin_df, imputed_df, missing_ind):
        """
        Compare values before missingles and after imputation:
        The best results here then value form origin Df equal to value in imputed DF
        """
        origin_val = origin_df.values.reshape(-1, )[missing_ind]
        imputed_val = imputed_df.values.reshape(-1, )[missing_ind]
        return self.rmse(origin_val, imputed_val)

    def plot_heatmap(self, score_data: pd.DataFrame, mask_miss: np.array):
        mask = np.ones_like(score_data).reshape(-1, )
        mask[mask_miss] = False
        mask = mask.reshape(score_data.shape)
        _, ax = plt.subplots(figsize=(8, 8))
        with sns.axes_style("dark"):
            axs = sns.heatmap(score_data, annot=True, mask=mask, square=True, ax=ax, cmap='Reds')
            axs.set_title(
                f'RMSE scores for metod: {score_data["name_imputer"].upper()}',
                f'Avg: {np.round(np.mean(score_data.values.reshape(-1, )),4)}'
            )

    def get_metric_score(self, origin_val, imputed_val, metric_name):
        pass

    def plot_metrics(self, scores_df):
        """
        Plot scores for all methods in scores_in_df.

        Where each row are scores for method in diff mode: clf, rgr, clastering

        param: scores_df: pd.DataFrame (columns:
                [method, score_clf, score_rgr, score_cls, diff_w_origin_cat, diff_w_origin_num])
        return: None
        """
        pass

    def get_metrics_unsupervised(self, df_origin, df_imputed):
        pass

    def get_metrics_supervised(self, df_origin, df_imputed, metric_type: str):
        """Metric_type Union[clf, regr]"""
        pass

    def f1_score(self, features, target):
        """
        Supervised classification score to compare diff imputers
        """
        clf = RandomForestClassifier().fit(features, target)
        # clf = LGBMClassifier().fit(feature, target)
        prediction = clf.predict(features)
        if isinstance(target, pd.Series):
            target = target.values
        # ohe = OneHotEncoder()
        # target = ohe.fit_transform(target.reshape(-1, 1)).toarray()
        res = f1_score(target, prediction, average='weighted', labels=np.unique(prediction))
        return res

    def silhouette(self, features):
        """
        Unsupervised clusterization score
        """
        claster_labels = AgglomerativeClustering().fit_predict(features)
        res = silhouette_score(features, claster_labels)
        return res
