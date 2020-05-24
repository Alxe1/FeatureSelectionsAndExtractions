# -*- coding:utf-8 -*-
# file       : relif.py
# time       : 2020/5/23 1:21 PM
# author     : littlely
# description:
import random

import numpy as np

from bases import Base


class Relif(Base):
    def __init__(self, max_iter, tao):
        """
        This is a simple implementation of relif algorithm which used for feature
        selections, the relif is simple to understand and high performance, but it
        can only deal with binary classification.

        Pay attention: relif use random sample selection of same class rather than
        using nearest neighbor sample to calculate nearest hit and miss, and it use
        number or string to make difference between named variable and numeric variable.
        it cannot handle null data, it will be improved later.

        Read more in :ref:`https://blog.csdn.net/littlely_ll/article/details/71614826`.

        :param max_iter: max iterations of relif

        :param tao: the threshold of feature weight
        """
        self.max_iter = max_iter
        self.tao = tao
        self.chosen_features = dict()

    def fit(self, X, y):
        """
        fit an array data

        :param X: a numpy array

        :param y: the label, a list or one dimension array

        :return:
        """
        assert isinstance(X, np.ndarray), "input should be an array!"

        if isinstance(y, list):
            y = np.array(y)

        assert X.shape[0] == len(y), "X and y not in the same length!"

        m, n = X.shape

        weight = np.zeros(n)

        for _ in range(self.max_iter):
            sample_seed = random.randint(0, m-1)
            sample_y = y[sample_seed]
            sample_x = X[sample_seed]

            while True:
                seed = random.randint(0, m - 1)
                if y[seed] == sample_y and sample_seed != seed:
                    near_hit = X[seed]
                    break
            while True:
                seed = random.randint(0, m - 1)
                if y[seed] != sample_y:
                    near_miss = X[seed]
                    break

            for i in range(n):
                near_hit_sum = 0
                near_miss_sum = 0
                if isinstance(sample_x[i], str):
                    if sample_x[i] != near_hit[i]:
                        near_hit_sum += 1
                    if sample_x[i] != near_miss[i]:
                        near_miss_sum += 1
                elif isinstance(sample_x[i], float):
                    near_hit_sum += pow(sample_x - near_hit, 2)
                    near_miss_sum += pow(sample_x - near_miss, 2)

                weight[i] += near_miss_sum - near_hit_sum

        weight = weight / self.max_iter

        for i, w in enumerate(weight):
            if w >= self.tao:
                self.chosen_features.update({i: w})

    def transform(self, X):
        """
        transform the array data

        :param X: array of data

        :return: the selected features
        """
        chosen_features = list(self.chosen_features.keys())
        return X[:, chosen_features]

    def fit_transform(self, X, y):
        self.fit(X, y)
        return self.transform(X)

    @property
    def important_features(self):
        return self.chosen_features
