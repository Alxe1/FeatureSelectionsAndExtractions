# -*- coding:utf-8 -*-
# file       : mrmr.py
# author     : littlely
# description:
import warnings

import numpy as np

from bases.base import Base


class MRMR(Base):
    def __init__(self, feature_num):
        """
        mRMR is a feature selection which maximises the feature-label correlation and minimises
        the feature-feature correlation. this implementation can only applied for numeric values,
        read more about mRMR, please refer :ref:`https://blog.csdn.net/littlely_ll/article/details/71749776`.

        :param feature_num: selected number of features
        """
        self.feature_num = feature_num
        self._selected_features = []

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

        if self.feature_num > X.shape[1]:
            self.feature_num = X.shape[1]
            warnings.warn("The feature_num has to be set less or equal to {}".format(X.shape[1]), UserWarning)

        MIs = self.feature_label_MIs(X, y)
        max_MI_arg = np.argmax(MIs)

        selected_features = []

        MIs = list(zip(range(len(MIs)), MIs))
        selected_features.append(MIs.pop(int(max_MI_arg)))

        while True:
            max_theta = float("-inf")
            max_theta_index = None

            for mi_outset in MIs:
                ff_mis = []
                for mi_inset in selected_features:
                    ff_mi = self.feature_feature_MIs(X[:, mi_outset[0]], X[:, mi_inset[0]])
                    ff_mis.append(ff_mi)
                theta = mi_outset[1] - 1 / len(selected_features) * sum(ff_mis)
                if theta >= max_theta:
                    max_theta = theta
                    max_theta_index = mi_outset
            selected_features.append(max_theta_index)
            MIs.remove(max_theta_index)

            if len(selected_features) >= self.feature_num:
                break

        self._selected_features = [ind for ind, mi in selected_features]

        return self

    def transform(self, X):
        return X[:, self._selected_features]

    def fit_transform(self, X, y):
        self.fit(X, y)
        return self.transform(X)

    def entropy(self, c):
        """
        entropy calculation

        :param c:

        :return:
        """
        c_normalized = c / float(np.sum(c))
        c_normalized = c_normalized[np.nonzero(c_normalized)]
        H = -sum(c_normalized * np.log2(c_normalized))
        return H

    def feature_label_MIs(self, arr, y):
        """
        calculate feature-label mutual information

        :param arr:

        :param y:

        :return:
        """
        m, n = arr.shape
        MIs = []
        p_y = np.histogram(y)[0]
        h_y = self.entropy(p_y)

        for i in range(n):
            p_i = np.histogram(arr[:, i])[0]
            p_iy = np.histogram2d(arr[:, 0], y)[0]

            h_i = self.entropy(p_i)
            h_iy = self.entropy(p_iy)

            MI = h_i + h_y - h_iy
            MIs.append(MI)
        return MIs

    def feature_feature_MIs(self, x, y):
        """
        calculate feature-feature mutual information

        :param x:

        :param y:

        :return:
        """
        p_x = np.histogram(x)[0]
        p_y = np.histogram(y)[0]
        p_xy = np.histogram2d(x, y)[0]

        h_x = self.entropy(p_x)
        h_y = self.entropy(p_y)
        h_xy = self.entropy(p_xy)

        return h_x + h_y - h_xy

    @property
    def important_features(self):
        return self._selected_features
