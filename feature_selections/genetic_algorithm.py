# -*- coding:utf-8 -*-
# file       : genetic_algorithm.py
# time       : 2020/5/28 8:23 PM
# author     : littlely
# description:
import math
import random
import warnings

import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import cross_validate

from bases.base import Base


class GA(Base):
    def __init__(self, size, feature_num, c_num, gamma_num, cv, svm_weight, feature_weight, C, keep_prob, cross_prob,
                 mutate_prob, iters, topK, topF):
        """
        This is the implementation of Genetic Algorithm, for more information please refer
        :ref:`https://blog.csdn.net/littlely_ll/article/details/72625312`

        :param size: the population size

        :param feature_num: the number of features which to choose from

        :param c_num: binary value length which value represent for SVM cost parameter

        :param gamma_num: binary value length which value represent for SVM gamma parameter

        :param cv: the k-fold which to train SVM

        :param svm_weight: SVM accuracy weight of fitness

        :param feature_weight: the features weight of fitness

        :param C: the features cost, which should be a number or numpy array, if numpy array, it must have the
                  same length with feature attributes

        :param keep_prob: the proportion of population

        :param cross_prob: the probability of gene cross

        :param mutate_prob: the probability of gene mutation

        :param iters: the iteration number of GA

        :param topK: select the best topK individuals to choose features

        :param topF: select number of features
        """
        self.size = size
        self.feature_num = feature_num
        self.c_num = c_num
        self.gamma_num = gamma_num
        self.cv = cv
        self.svm_weight = svm_weight
        self.feature_weight = feature_weight
        self.C = C
        self.keep_prob = keep_prob
        self.cross_prob = cross_prob
        self.mutate_prob = mutate_prob
        self.iters = iters
        self.topK = topK
        self.topF = topF

        self.average_fitness = []
        self.best_feature_index = None

    def fit(self, X, y):
        """
        fit the array data

        :param X: the numpy array

        :param y: the label, a list or one dimension array

        :return:
        """

        if all(np.array([self.svm_weight, self.feature_weight]) >= 0):
            if self.svm_weight + self.feature_weight > 1:
                self.svm_weight = self.svm_weight / (self.svm_weight + self.feature_weight)
                self.feature_weight = self.feature_weight / (self.svm_weight + self.feature_weight)

            assert self.svm_weight + self.feature_weight == 1.0, "svm_weight + feature_weight should be 1."
        else:
            raise ValueError("svm_weight and feature_weight argument can not be negative!")

        X = self._check_array(X)
        y = self._check_array(y)

        fitness_array = np.array([])

        population = self.generate_population(size=self.size, feature_num=self.feature_num, c_num=self.c_num,
                                              gamma_num=self.gamma_num)
        for _iter in range(self.iters):
            generators = self.features_and_params_generator(X, y, population=population, feature_num=self.feature_num,
                                                            c_num=self.c_num, gamma_num=self.gamma_num)
            fitness_list = []
            for i, (_features, _y, _c, _gamma) in enumerate(generators):
                svm_acc = GA.svm_classifier(_features, _y, _c, _gamma, self.cv)
                _fitness = self.fitness(population[i], svm_acc, self.svm_weight, self.feature_weight, C=self.C)
                fitness_list.append(_fitness)

            fitness_array = np.array(fitness_list)
            self.average_fitness.append(float(fitness_array.mean()))

            population = self.select_population(population, fitness_array, keep_prob=self.keep_prob)
            population = self.gene_cross(population, self.cross_prob)
            population = self.gene_mutate(population, self.mutate_prob)
            population = population[np.where(population[:, 0:self.feature_num].any(axis=1))[0], :]

        sorted_index = np.argsort(fitness_array)
        best_individuals = population[sorted_index[-self.topK:], :]

        feature_sum = best_individuals.sum(axis=1)
        sorted_feature_sum = np.argsort(feature_sum)
        self.best_feature_index = sorted(sorted_feature_sum[-self.topF:])

        return self

    def transform(self, X):
        X = self._check_array(X)
        return X[:, self.best_feature_index]

    def fit_transform(self, X, y):
        self.fit(X, y)
        return self.transform(X)

    @property
    def important_features(self):
        return self.best_feature_index

    def average_fitness(self):
        return self.average_fitness

    def fitness(self, gene, svm_acc, svm_weight, feature_weight, C):
        """ calculate the fitness of individuals """

        _fitness = svm_weight * svm_acc + feature_weight / float(sum(C * gene[0:self.feature_num]))
        return _fitness

    def select_population(self, population, fitness, keep_prob):
        """ select superior group with roulette """

        prob_fitness = fitness / np.sum(fitness)
        cumsum_fitness = np.cumsum(prob_fitness)
        rands = [random.random() for _ in range(len(fitness))]

        new_population_index = []
        for i, rand in enumerate(rands):
            for j, prob in enumerate(cumsum_fitness):
                if rand <= prob:
                    new_population_index.append(j)
                    break
        new_population_index = np.asarray(new_population_index)

        keep_population_num = math.ceil(len(new_population_index) * keep_prob)

        # the bigger the probability, the easier the individual to be get
        selected_population_index = np.random.choice(a=new_population_index, size=keep_population_num, replace=False)

        return population[selected_population_index, :]

    def gene_cross(self, population, cross_prob):
        """gene cross, if cross gene, it will choice two parents randomly, and generate two new generations"""

        gene_num = len(population[0])
        new_generation = np.zeros((1, gene_num))

        for i in range(len(population)):
            rand = random.random()
            if rand <= cross_prob:
                parents_index = np.random.choice(len(population), 2, replace=False)
                parents = population[parents_index, :]
                cross_point_1 = random.randint(0, gene_num - 2)
                cross_point_2 = random.randint(cross_point_1 + 1, gene_num)

                tmp = parents[0, cross_point_1:cross_point_2].copy()
                parents[1, cross_point_1:cross_point_2] = parents[0, cross_point_1:cross_point_2]
                parents[0, cross_point_1:cross_point_2] = tmp

                new_generation = np.concatenate([new_generation, parents], axis=0)
        if new_generation.any() == 0:
            warnings.warn("No cross in population!", UserWarning)
            return population
        else:
            new_generation = np.delete(new_generation, 0, axis=0)
            new_generation = np.concatenate([new_generation, population], axis=0)
            return new_generation

    def gene_mutate(self, population, mutate_prob):
        """mutate gene with specific probability, it will randomly choose split point"""

        population_size = len(population)
        gene_num = len(population[0])
        for i in range(population_size):
            rand = random.random()
            if rand <= mutate_prob:
                mutate_point = random.randint(0, gene_num - 1)
                if population[i, mutate_point] == 0:
                    population[i, mutate_point] = 1
                else:
                    population[i, mutate_point] = 0

        return population

    def generate_population(self, size, feature_num, c_num, gamma_num):
        """generate population with populatition size and feature size and parameter size"""

        features = np.array([[random.randint(0, 1) for _ in range(feature_num)] for _ in range(size)])
        params = np.array([[random.randint(0, 1) for _ in range(c_num + gamma_num)] for _ in range(size)])
        population = np.concatenate([features, params], axis=1)

        population = population[np.where(population[:, 0:feature_num].any(axis=1))[0], :]

        return population

    def features_and_params_generator(self, X, y, population, feature_num, c_num, gamma_num):
        """generate samples for each individuals"""

        feature_genes = population[:, 0:feature_num]
        C_genes = population[:, feature_num:feature_num + c_num].astype(str)
        gamma_genes = population[:, feature_num + c_num:feature_num + c_num + gamma_num].astype(str)

        for i in range(len(population)):
            _features = X[:, np.where(feature_genes[i] == 1)[0]]
            # specify the svm cost parameter range 0.001 to 100
            _C = GA.bin2int("".join(C_genes[i]), 0.001, 100)
            # specify the svm gamma parameter range 0.001 to 10
            _gamma = GA.bin2int("".join(gamma_genes[i]), 0.001, 10)
            yield _features, y, _C, _gamma

    @staticmethod
    def bin2int(value, low, high):
        """binary value to real value with a specific range"""

        gene_num = len(value)
        x = low + int(value, 2) * (high - low) / (pow(2, gene_num) - 1)
        return x

    @staticmethod
    def svm_classifier(X, y, cost, gamma, cv):
        svm = SVC(C=cost, kernel="rbf", gamma=gamma)
        cv_results = cross_validate(svm, X, y, scoring="accuracy", cv=cv)
        accuracy = float(cv_results['test_score'].mean())

        return accuracy
