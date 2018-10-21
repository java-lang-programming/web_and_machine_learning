# coding: utf-8
from dataset import aclImdb
from dataset import stopword
import numpy as np
import sys
sys.path.append('..')
import re
from nltk.stem.porter import PorterStemmer

# http://scikit-learn.org/stable/modules/sgd.html
# http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html
class LinearRegressionGD:
    def __init__(self, eta=0.001, n_iter=20):
        self._eta = eta
        self._n_iter = n_iter

    def fit(self, X, y):
        self._w = np.zeros(1 + X.shape[1])
        self._cost = []
        for i in range(self._n_iter):
            output = self.net_input(X)
            #print(output)
            errors = (y - output)
            #print(errors)
            self._w[1:] += self._eta * X.T.dot(errors)
            self._w[0] += self._eta * errors.sum()
            cost = (errors**2).sum() / 2.0
            self._cost.append(cost)
        return self

    def net_input(self, X):
        return np.dot(X, self._w[1:]) + self._w[0]

    def predict(self, X):
        return self.net_input(X)
