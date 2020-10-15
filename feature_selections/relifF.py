# -*- coding:utf-8 -*-
# file       : relifF.py
# time       : 2020/5/23 9:36 PM
# author     : littlely
# description:
import random

import numpy as np

from bases.base import Base


class RelifF(Base):
    def __init__(self, max_iter, tao, neighbors):
        """
        This is a simple implementation of relifF algorithm which used for feature
        selections, the relifF is simple to understand and can process multi-classifications.

        Pay attention: relifF use random sample selection of same class rather than
        using nearest neighbor sample to calculate nearest hit and miss, and it cannot handle
        null data, it will be improved later.

        Read more in :ref:`https://blog.csdn.net/littlely_ll/article/details/71614826`.

        :param max_iter: max iterations of relifF

        :param tao: the threshold of feature weight

        :param neighbors: the neighbors of each class to calculate weight
        """
        self.max_iter = max_iter
        self.tao = tao
        self.neighbors = neighbors
        self._weight = None
        self._important_weight = dict()

    def fit(self, X, y):
        """
        fit an array data

        :param X: a numpy array

        :param y: the label, a list or one dimension array

        :return:
        """
        X = self._check_array(X)
        y = self._check_array(y)
        assert X.shape[0] == len(y), "X and y not in the same length!"

        m, n = X.shape

        self._weight = np.zeros(n)

        label_count = dict()
        label_index = dict()
        for label in np.unique(y):
            label_index[label] = np.where(y == label)[0]
            label_count[label] = len(np.where(y == label)[0])

        # label probability
        label_probability = dict((label, count/m) for label, count in label_count.items())

        col_type = []
        for i in range(n):
            if isinstance(X[:,i][0], str):
                col_type.append((1,))
            else:
                col_min = X[:, i].min()
                col_max = X[:, i].max()
                difference = col_max - col_min
                col_type.append((0, difference))

        for _ in range(self.max_iter):
            sample_seed = random.randint(0, m - 1)
            sample_y = y[sample_seed]
            sample_x = X[sample_seed]

            for j in range(n):
                near_hit_sum = 0
                near_miss_sum = 0
                for label in label_index.keys():
                    if label == sample_y:
                        near_hit_neighbors = np.random.choice(label_index[label], self.neighbors, replace=False)
                        for i in near_hit_neighbors:
                            sample_i = X[i]
                            if col_type[j][0] == 1:
                                if sample_x[j] != sample_i[j]:
                                    near_hit_sum += 1
                            else:
                                near_hit_sum += np.abs(sample_x[j] - sample_i[j]) / col_type[j][1]
                    else:
                        pre_near_miss_sum = 0
                        near_miss_neighbors = np.random.choice(label_index[label], self.neighbors, replace=False)
                        for i in near_miss_neighbors:
                            sample_i = X[i]
                            if col_type[j][0] == 1:
                                if sample_x[j] != sample_i[j]:
                                    pre_near_miss_sum += 1
                            else:
                                pre_near_miss_sum += np.abs(sample_x[j] - sample_i[j]) / col_type[j][1] + 0.001
                        near_miss_sum += pre_near_miss_sum * label_probability[label] / (
                                    1 - label_probability[sample_y])

                self._weight[j] += (near_miss_sum - near_hit_sum) / self.neighbors

        for i, w in enumerate(self._weight):
            if w >= self.tao:
                self._important_weight[i] = w

    def transform(self, X):
        """
        transform the array data

        :param X:  array of data

        :return: the selected features
        """
        important_col = list(self._important_weight.keys())
        return X[:, important_col]

    def fit_transform(self, X, y):
        self.fit(X, y)
        return self.transform(X)

    @property
    def important_features(self):
        return self._important_weight

    @property
    def weight(self):
        return self._weight
