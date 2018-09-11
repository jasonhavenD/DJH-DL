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
import tensorflow as tf

w = tf.Variable(tf.constant(5,dtype=tf.float32))

loss = tf.square(w + 1)

train_step = tf.train.GradientDescentOptimizer(0.2).minimize(loss)

with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())
	for i in range(40):
		sess.run(train_step)
		w_val = sess.run(w)
		loss_val = sess.run(loss)

		print('w is %f , loss is %f.' % (w_val, loss_val))
