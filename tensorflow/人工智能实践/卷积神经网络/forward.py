#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name:forward
   Author:jasonhaven
   date:18-9-11
-------------------------------------------------
   Change Activity:18-9-11:
-------------------------------------------------
"""
import tensorflow as tf

IMAGE_SIZE = 28
NUM_CHANNELS = 1
CONV1_SIZE = 5
CONV1_KERNEL_NUM = 32

CONV2_SIZE = 5
CONV2_KERNEL_NUM = 64

FC_SIZE = 512
OUTPUT_NODE = 10


def get_weight(shape, regularizer):
	W = tf.Variable(tf.truncated_normal(shape, stddev=0.1), dtype=tf.float32)
	if regularizer:
		tf.add_to_collection('losses', tf.contrib.layers.l2_regularizer(regularizer)(W))
	return W


def get_bias(shape):
	b = tf.Variable(tf.zeros(shape), dtype=tf.float32)
	return b


def conv2d(x, W, strides=[1, 1, 1, 1]):
	return tf.nn.conv2d(x, W, strides=strides, padding='SAME')


def max_pool_2x2(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1]):
	return tf.nn.max_pool(x, ksize=ksize, strides=strides, padding='SAME')


def forward(x, is_train=False, regularizer=None):
	conv1_w = get_weight([CONV1_SIZE, CONV1_SIZE, NUM_CHANNELS, CONV1_KERNEL_NUM], regularizer)
	conv1_b = get_bias([CONV1_KERNEL_NUM])
	conv1 = conv2d(x, conv1_w)
	relu1 = tf.nn.relu(tf.nn.bias_add(conv1, conv1_b))
	pool1 = max_pool_2x2(relu1)

	conv2_w = get_weight([CONV2_SIZE, CONV2_SIZE, CONV1_KERNEL_NUM, CONV2_KERNEL_NUM], regularizer)
	conv2_b = get_bias([CONV2_KERNEL_NUM])
	conv2 = conv2d(pool1, conv2_w)
	relu2 = tf.nn.relu(tf.nn.bias_add(conv2, conv2_b))
	pool2 = max_pool_2x2(relu2)

	pool_shape = pool2.get_shape().as_list()
	nodes = pool_shape[1] * pool_shape[2] * pool_shape[3]
	reshaped = tf.reshape(pool2, [pool_shape[0], nodes])

	fc1_w = get_weight([nodes, FC_SIZE], regularizer)
	fc1_b = get_bias([FC_SIZE])
	fc1 = tf.nn.relu(tf.matmul(reshaped, fc1_w) + fc1_b)
	if is_train: fc1 = tf.nn.dropout(fc1, 0.5)

	fc2_w = get_weight([FC_SIZE, OUTPUT_NODE], regularizer)
	fc2_b = get_bias([OUTPUT_NODE])
	y = tf.matmul(fc1, fc2_w) + fc2_b
	return y
