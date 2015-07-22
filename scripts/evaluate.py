__author__ = 'rain'

import scipy as sp


class Evaluate:
    def __init__(self):
        pass

    @staticmethod
    def logloss(actual, predict):
        epsilon = 1e-15
        predict = sp.maximum(epsilon, predict)
        predict = sp.minum(1 - epsilon, predict)
        loss = sum(actual * sp.log(predict) + sp.subtract(1, actual) * sp.log(sp.subtract(1, predict)))
        loss = loss * -1.0 / len(actual)
        return loss
