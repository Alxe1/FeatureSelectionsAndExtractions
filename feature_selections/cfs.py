# -*-coding: utf-8-*-

# Author     : LiuLei 
# FileName   : cfs.py
# DateTime   : 2020/5/25 15:14
# Description:
from itertools import combinations

import numpy as np

from bases import Base


class CFS(Base):
    def __init__(self):
        self._relavent_cols = []
        self._merits = None

    def fit(self, X, y):
        if isinstance(y, list):
            y = np.asarray(y)
        assert isinstance(X, np.ndarray), "X should be an array!"
        assert isinstance(y, np.ndarray), "y should be a list or an array!"
        assert len(X) == len(y), "X and y should have same length!"

        m, n = X.shape

        for i in range(n):
            if np.var(X[:, i]) == 0.0:
                raise ValueError("Column feature should not be zero variance!")
            if isinstance(X[:, i][0], str):
                raise ValueError("It does not support string values yet!")

        correlations = np.corrcoef(X, y, rowvar=False)
        correlations = correlations[:-1, :]

        _max_index = np.argmax(correlations[:, -1])
        self._relavent_cols.append(_max_index)
        self._merits = correlations[_max_index, -1]

        while True:
            _tmp_relavent = []
            tmp_relavent_col = None
            max_merits = float("-inf")

            for i in range(n):
                if i not in self._relavent_cols:
                    _tmp_relavent.extend(self._relavent_cols)
                    _tmp_relavent.append(i)
                    row_ind, col_ind = zip(*combinations(_tmp_relavent, 2))

                    ff_mean = correlations[row_ind, col_ind].mean()
                    fc_mean = correlations[_tmp_relavent, -1].mean()

                    k = len(_tmp_relavent)
                    merits = (k * fc_mean) / np.sqrt(k + k*(k-1)*ff_mean)
                    if merits >= max_merits:
                        max_merits = merits
                        tmp_relavent_col = i

            if max_merits > self._merits:
                self._relavent_cols.append(tmp_relavent_col)
                self._merits = max_merits
            else:
                break

    def transform(self, X):
        return X[:, self._relavent_cols]

    def fit_transform(self, X, y):
        self.fit(X, y)
        return self.transform(X)

    @property
    def merits(self):
        return self._merits

    @property
    def important_features(self):
        return self._relavent_cols