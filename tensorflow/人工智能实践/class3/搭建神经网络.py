#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name:搭建神经网络
   Author:jasonhaven
   date:18-9-9
-------------------------------------------------
   Change Activity:18-9-9:
-------------------------------------------------
"""

#################################################
# 基本概念

'''
shape 不能使用数值,使用int32数组[1],[1,2,3]...
'''

#################################################
import tensorflow as tf
import numpy as np

t = tf.constant(0, dtype=tf.int32)
print(t)  # Tensor("Const:0", shape=(), dtype=int32)

t = tf.constant([1, 2], dtype=tf.float32)
print(t)  # Tensor("Const_1:0", shape=(2,), dtype=float32)

t = tf.constant([1, 2, 3], dtype=tf.float32, name='my_a')
print(t)  # Tensor("my_a:0", shape=(3,), dtype=float32)

x = tf.constant([[1.0, 2.0]])  # 定义一个 2 阶张量等于[[1.0,2.0]] 1*2
w = tf.constant([[3.0], [4.0]])  # 定义一个 2 阶张量等于[[3.0],[4.0]]  2*1
y = tf.matmul(x, w)  # 实现 xw 矩阵乘法 1*1
print(y)  # 1*1

with tf.Session() as sess:
	print(sess.run(y))  # [[11.]]

#################################################
# 神经网络的参数
#################################################

'''
生成随机数/数组的函数

seed 

随机种子如果去掉每次生成的随机数将不一致
如果没有特殊要求，随机种子可以不写


'''
# 正态分布
print(tf.random_normal([1, 2], mean=0, stddev=1, dtype=tf.float32, seed=1))
# 去掉偏离过大的正态分布
print(tf.truncated_normal([1, 2], mean=0, stddev=1, dtype=tf.float32, seed=1))
# 随机均匀分布
print(tf.random_uniform([2, 3], minval=-1, maxval=1, dtype=tf.float32, seed=1))

'''
Tensors常量值函数
'''
shape = [1, 2]
value = 0
dims = [1, 2]
tensor = tf.Variable([1, 1], dtype=tf.float64)

tf.zeros(shape, dtype=tf.float32, name=None)
tf.zeros_like(tensor, dtype=None, name=None)
tf.ones(shape, dtype=tf.float32, name=None)
tf.ones_like(tensor, dtype=None, name=None)
tf.fill(dims, value, name=None)  # int32, int64
'''
创建一个维度为dims，值为value的tensor对象．
该tensor对象中的值类型和value一致

当value为０时，该方法等同于tf.zeros()
当value为１时，该方法等同于tf.ones()
参数:
dims: 类型为int32的tensor对象，用于表示输出的维度(1-D, n-D)，通常为一个int32数组，如：[1], [2,3]等
value: 常量值(字符串，数字等)，该参数用于设置到最终返回的tensor对象值中
name: 当前操作别名(可选)
'''

tf.constant(value, dtype=None, shape=None, name='Const')

#################################################
# 实现一个多组二维数据的两层感知机网络结构
#################################################

BATCH_SIZE = 8
nb_samples = 100
STEPS = 1000

# 生成数据
X = np.random.randn(nb_samples, 1)
print(X.shape)
Y = X * 0.5 + np.random.randn(nb_samples, 1) * 0.1
print(Y.shape)

# 查看数据
# import matplotlib.pyplot as plt
# plt.scatter(X, Y)
# plt.show()


# 定义输入和参数
x = tf.placeholder(tf.float32, [None, 1])
y_ = tf.placeholder(tf.float32, [None, 1])

W = tf.Variable(tf.random_normal([1, 5], dtype=tf.float32))
b = tf.Variable(tf.ones(1), dtype=tf.float32)

# 定义前向传播
y = tf.matmul(x, W) + b

# 定义反向传播
loss_mse = tf.reduce_mean(tf.square(y_ - y))
train_step = tf.train.GradientDescentOptimizer(0.02).minimize(loss_mse)

init_op = tf.global_variables_initializer()

# 会话
with tf.Session() as sess:
	sess.run(init_op)

	for step in range(STEPS + 1):
		start = (step * BATCH_SIZE) % nb_samples
		end = start + BATCH_SIZE
		sess.run(train_step, feed_dict={x: X[start:end], y_: Y[start:end]})
		if step % 100 == 0:
			print("After %d training step(s), loss_mse on all data is %g" % (
				step,
				sess.run(loss_mse, feed_dict={x: X, y_: Y})
			))
