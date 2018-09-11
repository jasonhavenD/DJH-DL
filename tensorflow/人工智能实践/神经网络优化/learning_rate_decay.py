#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name:learning_rate
   Author:jasonhaven
   date:18-9-10
-------------------------------------------------
   Change Activity:18-9-10:
-------------------------------------------------
"""

'''
#使用指数衰减的学习率，在迭代初期得到较高的下降速度，可以在较小的训练轮数下取得更有收敛度。
'''
import tensorflow as tf

# 定义参数
LEARNING_RATE_BASE = 0.5
LEARNING_RATE_DECAY = 0.99
LEARNING_RATE_STEP = 1
global_step = tf.Variable(0, trainable=False)

learning_rate = tf.train.exponential_decay(LEARNING_RATE_BASE,
                                           global_step,
                                           LEARNING_RATE_STEP,
                                           LEARNING_RATE_DECAY,
                                           staircase=True)

w = tf.Variable(tf.constant(5, dtype=tf.float32))

loss = tf.square(w + 1)

train_step = tf.train.GradientDescentOptimizer(0.2).minimize(loss)

with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())
	for i in range(40):
		sess.run(train_step)
		w_val = sess.run(w)
		loss_val = sess.run(loss)

		print('w is %f , loss is %f.' % (w_val, loss_val))
