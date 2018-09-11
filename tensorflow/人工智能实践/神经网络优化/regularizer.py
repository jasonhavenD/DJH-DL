#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name:regularizer
   Author:jasonhaven
   date:18-9-10
-------------------------------------------------
   Change Activity:18-9-10:
-------------------------------------------------
"""
# import
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

BATCH_SIZE = 32
STEPS = 40000
SEED = 2
rdm = np.random.RandomState(SEED)

# 随机数返回300行2列的矩阵，表示300组坐标点（x0,x1）作为输入数据集
X = rdm.randn(300, 2)
# 从X这个300行2列的矩阵中取出一行,判断如果两个坐标的平方和小于2，给Y赋值1，其余赋值0
# 作为输入数据集的标签（正确答案）
Y = [int(x1 * x1 + x2 * x2 < 2) for (x1, x2) in X]

# 遍历Y中的每个元素，1赋值'red'其余赋值'blue'，这样可视化显示时人可以直观区分
Y_c = ['red' if y == 1 else 'blue' for y in Y]

# 对数据集X和标签Y进行shape整理，第一个元素为-1表示，随第二个参数计算得到，
# 第二个元素表示多少列，把X整理为n行2列，把Y整理为n行1列
X = np.vstack(X).reshape(-1, 2)
Y = np.vstack(Y).reshape(-1, 1)

print(X.shape, Y.shape)

# draw plot
plt.scatter(X[:, 0], X[:, 1], c=Y_c)
plt.show()


# 定义神经网络的输入，参数，输出
def get_weight(shape, regularizer):
	W = tf.Variable(tf.random_normal(shape, dtype=tf.float32))
	tf.add_to_collection('losses', tf.contrib.layers.l2_regularizer(regularizer)(W))
	return W


# print(get_weight([2, 2], 0.2))

def get_bias(shape):
	b = tf.Variable(tf.random_normal(shape, dtype=tf.float32))
	return b


# print(get_bias([1, 2]))

# 定义前向
x = tf.placeholder(tf.float32, shape=[None, 2])
y_ = tf.placeholder(tf.float32, shape=[None, 1])

w1 = get_weight([2, 11], 0.01)
b1 = get_bias([11])
y1 = tf.nn.relu(tf.matmul(x, w1) + b1)

w2 = get_weight([11, 1], 0.01)
b2 = get_bias([1])
y = tf.matmul(y1, w2) + b2

# 定义反向
loss_mse = tf.reduce_mean(tf.square(y_ - y))
loss_total = loss_mse + tf.add_n(tf.get_collection('losses'))

# 不包含正则化
train_step = tf.train.AdamOptimizer(0.001).minimize(loss_mse)

with tf.Session() as sess:
	init_op = tf.global_variables_initializer()
	sess.run(init_op)
	for i in range(STEPS):
		start = (i * BATCH_SIZE) % 300
		end = start + BATCH_SIZE
		sess.run(train_step, feed_dict={x: X[start:end], y_: Y[start:end]})
		if i % 2000 == 0:
			loss_mse_v = sess.run(loss_mse, feed_dict={x: X, y_: Y})
			print("After %d steps, loss is: %f" % (i, loss_mse_v))
	# xx在-3到3之间以步长为0.01，yy在-3到3之间以步长0.01,生成二维网格坐标点
	xx, yy = np.mgrid[-3:3:.01, -3:3:.01]
	# 将xx , yy拉直，并合并成一个2列的矩阵，得到一个网格坐标点的集合
	grid = np.c_[xx.ravel(), yy.ravel()]
	# 将网格坐标点喂入神经网络 ，probs为输出
	probs = sess.run(y, feed_dict={x: grid})
	# probs的shape调整成xx的样子
	probs = probs.reshape(xx.shape)

plt.scatter(X[:, 0], X[:, 1], c=Y_c)
plt.contour(xx, yy, probs, levels=[.5])
plt.show()

# 定义反向传播方法：包含正则化
train_step = tf.train.AdamOptimizer(0.0001).minimize(loss_total)

with tf.Session() as sess:
	init_op = tf.global_variables_initializer()
	sess.run(init_op)
	for i in range(STEPS):
		start = (i * BATCH_SIZE) % 300
		end = start + BATCH_SIZE
		sess.run(train_step, feed_dict={x: X[start:end], y_: Y[start:end]})
		if i % 2000 == 0:
			loss_mse_v = sess.run(loss_mse, feed_dict={x: X, y_: Y})
			print("After %d steps, loss is: %f" % (i, loss_mse_v))

	xx, yy = np.mgrid[-3:3:.01, -3:3:.01]
	grid = np.c_[xx.ravel(), yy.ravel()]
	probs = sess.run(y, feed_dict={x: grid})
	probs = probs.reshape(xx.shape)

plt.scatter(X[:, 0], X[:, 1], c=Y_c)
plt.contour(xx, yy, probs, levels=[.5])
plt.show()
