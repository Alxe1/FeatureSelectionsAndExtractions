# -*- coding:utf-8 -*-
# file       : bases.py
# time       : 2020/5/23 9:50 PM
# author     : littlely
# description: 


class Base(object):

    def fit(self, X, y):
        raise NotImplementedError

    def transform(self, X):
        raise NotImplementedError

    def fit_transform(self, X, y):
        raise NotImplementedError
