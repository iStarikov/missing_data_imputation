import numpy as np
import pandas as pd
from fancyimpute import KNN, IterativeSVD, MatrixFactorization, IterativeImputer
from gensim.models import FastText
from statsmodels.imputation import mice
import tensorflow as tf

import utilities


class ImputationMaster:

    def __init__(self, strategy_name: str):
        self.strategy_name = strategy_name

    def complete(self, data):
        """
        Impute all missing data with class model
        :pram: data - pd.DataFrame
        :return: pd.DataFrame
        """
        raise NotImplementedError("Derived class needs to implement complete.")


# todo: add test-train split
class DummyImputation(ImputationMaster):
    fill_dict = None

    STRATEGIES = {
        'median': np.nanmedian,
        'mean': np.nanmean,
        'mode': utilities.mode,
        'constant': utilities.constant
    }

    def __init__(self, strategy_name: str):
        super().__init__(strategy_name)
        self.strategy = self.STRATEGIES[strategy_name]

    def complete(self, df_in: pd.DataFrame):
        self.fit(df_in)
        df = self.transform(df_in)
        return df

    def fit(self, df_in: pd.DataFrame):
        self.fill_dict = {}
        for j in df_in.columns:
            self.fill_dict[j] = self.strategy(df_in[j])

    def transform(self, df_in):
        df = df_in.copy()
        df = df.fillna(self.fill_dict)
        return df


class MiceImputation(ImputationMaster):
    def __init__(self, strategy_name: str = 'mice'):
        super().__init__(strategy_name)

    def complete(self, data: pd.DataFrame):
        df = data.copy()
        imp = mice.MICEData(df)
        p = {}
        for i in range(0, 20):
            p[i] = imp.next_sample()
        p = pd.Panel(p)
        df = p.mean(axis=0)
        return df


class IterativeImputation(ImputationMaster):
    def __init__(self, strategy_name: str = 'mice', n_iter: int = None):
        super().__init__(strategy_name)
        self.n_iter = n_iter

    def complete(self, data: pd.DataFrame):
        df = data.copy()
        cols = list(df)
        df = pd.DataFrame(IterativeImputer(n_iter=self.n_iter, verbose=False).fit_transform(df))
        df.columns = cols
        return df


class KNNImputation(ImputationMaster):
    def __init__(self, strategy_name: str = 'knn', k: int = None):
        super().__init__(strategy_name)
        self.k = k

    def complete(self, data: pd.DataFrame):
        df = data.copy()
        cols = list(df)
        df = pd.DataFrame(KNN(k=self.k, verbose=False).fit_transform(df))
        df.columns = cols
        return df


class SVDImputation(ImputationMaster):
    def __init__(self, strategy_name: str = 'svd', rank: int = None):
        super().__init__(strategy_name)
        self.rank = rank

    def complete(self, data: pd.DataFrame):
        df = data.copy()
        cols = list(df)
        if np.argmax(cols) < self.rank:
            self.rank = np.argmax(cols)
        df = pd.DataFrame(IterativeSVD(rank=self.rank, verbose=False).fit_transform(df))
        df.columns = cols
        return df


class MFImputation(ImputationMaster):
    def __init__(self, strategy_name: str = 'mf', rank: int = None):
        super().__init__(strategy_name)
        self.rank = rank

    def complete(self, data: pd.DataFrame):
        df = data.copy()
        cols = list(df)
        if np.argmax(cols) < self.rank:
            self.rank = np.argmax(cols)
        df = pd.DataFrame(MatrixFactorization(rank=self.rank, verbose=False).fit_transform(df))
        df.columns = cols
        return df


class AEImputation(ImputationMaster):
    def __init__(self, strategy_name: str = 'ae'):
        super().__init__(strategy_name)

    def complete(self, data: pd.DataFrame):
        df = data.copy()
        cols = list(df)
        df = self.run(df)
        df.columns = cols
        return df

    def run(self, df: pd.DataFrame):
        """
        Run AE model to learn recover missing data
        :param df: DF with missing data
        :return:
        """
        na_loc = df.isnull()
        df.fillna(df.mean(axis=0), inplace=True)
        # 7 as in article
        phi = 7
        data_shape = df.shape[1]
        x = tf.placeholder(shape=[None, data_shape], dtype=tf.float32)

        # model
        nn = tf.layers.dense(x, data_shape + phi, activation=tf.nn.tanh)
        nn = tf.layers.dense(nn, data_shape + 2 * phi, activation=tf.nn.tanh)
        encoded = tf.layers.dense(nn, data_shape + 3 * phi, activation=tf.nn.tanh)
        nn = tf.layers.dense(encoded, data_shape + 2 * phi, activation=tf.nn.tanh)
        # nn = tf.layers.dense(nn, data_shape+phi, activation=tf.nn.tanh)
        nn = tf.layers.dense(nn, data_shape, activation=tf.nn.tanh)
        cost = tf.reduce_mean((nn - x) ** 2)
        # todo: try Adam
        optimizer = tf.train.MomentumOptimizer(0.05, use_nesterov=True, momentum=0.99).minimize(cost)
        init = tf.global_variables_initializer()
        feed_df = df.copy()
        with tf.Session() as sess:
            sess.run(init)
            # 500 epochs in article
            for step in range(700):
                # corrupt part of data randomly to learn AE recover corrupted data
                x_feed = utilities.stochastic_corruption(feed_df)
                _, val = sess.run([optimizer, cost],
                                  feed_dict={x: x_feed})
                # if step % 100 == 0:
                #     print("step: {}, value: {}".format(step, val))
            features_imputed = sess.run([nn], feed_dict={x: df.values})
        tf.reset_default_graph()

        features_imputed = features_imputed[0]
        features_imputed = pd.DataFrame(features_imputed, columns=list(df))
        # impute missing data with data from last layer of AE
        df.update(features_imputed[na_loc])
        return df


class FastTextEmbeddingsImputation(ImputationMaster):
    def __init__(self, strategy_name: str = 'emb'):
        super().__init__(strategy_name)

    def complete(self, data: pd.DataFrame):
        df = data.copy()
        df = self.run(df)
        return df

    def run(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transform dataframe with missing values to DF of aggregated embeddings for each sample
        the final embeddings for sample is mean embedding vector of all embeddings vector of non missing values

        :param df: DF with missing data
        :return: embeddings DF
        """
        # encode features to string
        df_intrv = utilities.intervals(df, inplace=False)
        # concatenate values to sequence
        seq_df = utilities.concat_columns(df_intrv)
        # min_n=3, max_n=2 to avoid char gram
        size = df.shape[1]
        emb_size = 50
        window = 30
        if size < 60:
            emb_size = int(size - (1 + ((size // 10) ** 1.5 + (size // 10))))
            window = int(size - ((size // 10) ** 2 + (size // 10)))

        model = FastText(size=emb_size, window=window, min_count=1, workers=-1, min_n=3, max_n=2)  # instantiate
        model.build_vocab(sentences=seq_df)
        model.train(sentences=seq_df, total_examples=len(seq_df), epochs=50)
        # for high missingness level some rows don't include values impute most frequence in that case
        idxmax = df_intrv.apply(lambda col: col.value_counts()[0], axis=0).idxmax()
        idxmax = df_intrv[idxmax].value_counts().idxmax()
        seq_df = pd.Series([x if len(x) > 0 else [idxmax] for x in seq_df])
        # get embeddings sequences
        emb = seq_df.apply(lambda row: model.wv[row].mean(axis=0).tolist())
        # create DF with emb dims as columns
        emb_df = pd.DataFrame(np.row_stack(emb))
        emb_df.columns = ['x' + str(x) for x in emb_df.columns]
        return emb_df
