# coding:utf-8
# 0导入模块 ，生成模拟数据集
import numpy as np
import matplotlib.pyplot as plt

seed = 2


def generateds(nb_samples=300, nb_features=2):
	rdm = np.random.RandomState(seed)
	X = rdm.randn(nb_samples, nb_features)
	Y = [int(x1 * x1 + x2 * x2) for (x1, x2) in X]
	Y_c = ['red' if y else 'blue' for y in Y]

	X = np.vstack(X).reshape(-1, nb_features)
	Y = np.vstack(Y).reshape(-1, 1)
	return X, Y, Y_c
