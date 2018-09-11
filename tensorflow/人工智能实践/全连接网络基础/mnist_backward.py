#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name:mnist_backward
   Author:jasonhaven
   date:18-9-11
-------------------------------------------------
   Change Activity:18-9-11:
-------------------------------------------------
"""
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.keras.utils import to_categorical
import mnist_forward
import os

BATCH_SIZE = 64
# 衰减学习率
LEARNING_RARE_BASE = 0.1
LEARNING_RARE_DECAY = 0.99
# 正则化参数
REGULARIZER = 0.0001
# 迭代轮数
STEPS = 50000
# 滑动平均
MOVING_AVERAGE_DECAY = 0.99
# 保存路经
MODEL_SAVE_PATH = './model/'
MODEL_NAME = 'mnist_model'


def backward(mnist):
	x = tf.placeholder(tf.float32, [None, mnist_forward.INPUT_NODE])
	y_ = tf.placeholder(tf.float32, [None, mnist_forward.OUTPUT_NODE])
	y = mnist_forward.forward(x, REGULARIZER)
	global_step = tf.Variable(0, trainable=False)

	ce = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=tf.argmax(y_, 1), logits=y)
	cem = tf.reduce_mean(ce)
	loss = cem + tf.add_n(tf.get_collection('losses'))

	learning_rate = tf.train.exponential_decay(
		LEARNING_RARE_BASE,
		global_step,
		mnist.train.num_examples / BATCH_SIZE,
		LEARNING_RARE_DECAY,
		staircase=True)
	train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss,global_step=global_step)

	ema = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)
	ema_op = ema.apply(tf.trainable_variables())
	with tf.control_dependencies([train_step, ema_op]):
		train_op = tf.no_op(name='train')

	saver = tf.train.Saver()

	with tf.Session() as sess:
		init_op = tf.global_variables_initializer()
		sess.run(init_op)

		for i in range(STEPS):
			xs, ys = mnist.train.next_batch(BATCH_SIZE)
			assert xs.shape == (BATCH_SIZE, mnist_forward.INPUT_NODE)
			assert ys.shape == (BATCH_SIZE, mnist_forward.OUTPUT_NODE)
			_, loss_value, step = sess.run([train_op, loss, global_step], feed_dict={x: xs, y_: ys})
			if i % 1000 == 0:
				print("After %d training step(s), loss on training batch is %g." % (step, loss_value))
				saver.save(sess, os.path.join(MODEL_SAVE_PATH, MODEL_NAME), global_step)


def main():
	mnist_dir = '/home/jasonhaven/Downloads/NLP-DataSets/MNIST_data'
	mnist = input_data.read_data_sets(mnist_dir,one_hot=True)
	backward(mnist)


if __name__ == '__main__':
	main()
