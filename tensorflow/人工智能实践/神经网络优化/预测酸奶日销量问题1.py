#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name:预测酸奶日销量问题1
   Author:jasonhaven
   date:18-9-9
-------------------------------------------------
   Change Activity:18-9-9:
-------------------------------------------------
"""
'''
预测酸奶日销量 y， x1 和 x2 是影响日销量的两个因素。

假设x1和x2对销量的影响是一样的！

应提前采集的数据有：一段时间内，每日的 x1 因素、 x2 因素和销量 y_。采集的数据尽量多。
'''
# 导入模块
import tensorflow as tf
import numpy as np

# 定义参数
BATCH_SIZE = 8
SEED = 12345
nb_samples = 1000
STEPS = 2000

# 准备数据集
rdm = np.random.RandomState(SEED)
X = rdm.rand(nb_samples, 2)
Y = [[x1 + x2 + rdm.rand() / 10 - 0.5] for (x1, x2) in X]

# 定义输入
x = tf.placeholder(tf.float32, [None, 2])
y_ = tf.placeholder(tf.float32, [None, 1])
W1 = tf.Variable(tf.random_normal([2, 1], dtype=tf.float32))
b = tf.Variable(tf.ones([1]))

# 定义前向
y = tf.matmul(x, W1) + b

# 定义后向
loss_mse = tf.reduce_mean(tf.square(y_ - y))
train_step = tf.train.GradientDescentOptimizer(0.02).minimize(loss_mse)

init_op = tf.global_variables_initializer()

# 会话
with tf.Session() as sess:
	sess.run(init_op)
	print("Initial w1 is: \n", sess.run(W1))
	print("Initial b is: \n", sess.run(b))

	for step in range(STEPS + 1):
		start = (step * BATCH_SIZE) % nb_samples
		end = start + BATCH_SIZE
		sess.run(train_step, feed_dict={x: X[start:end], y_: Y[start:end]})
		if step % 100 == 0:
			print("After %d training step(s), loss_mse on all data is %g" % (
				step,
				sess.run(loss_mse, feed_dict={x: X, y_: Y})
			))
	print("Final w1 is: \n", sess.run(W1))
	print("Final b is: \n", sess.run(b))