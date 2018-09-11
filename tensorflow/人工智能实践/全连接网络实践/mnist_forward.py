#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name:mnist_forward
   Author:jasonhaven
   date:18-9-11
-------------------------------------------------
   Change Activity:18-9-11:
-------------------------------------------------
"""
import tensorflow as tf

INPUT_NODE = 784
OUTPUT_NODE = 10
LAYER1_NODE = 512


def get_weights(shape, regularizer):
	W = tf.Variable(tf.truncated_normal(shape, dtype=tf.float32,stddev=0.1))
	if regularizer:
		tf.add_to_collection('losses', tf.contrib.layers.l2_regularizer(regularizer)(W))
	return W


def get_bias(shape):
	b = tf.Variable(tf.zeros(shape, dtype=tf.float32))
	return b


def forward(x, regularizer):
	w1 = get_weights([INPUT_NODE, LAYER1_NODE], regularizer)
	b1 = get_bias([LAYER1_NODE])
	y1 = tf.nn.relu(tf.matmul(x, w1) + b1)

	w2 = get_weights([LAYER1_NODE, OUTPUT_NODE], regularizer)
	b2 = get_bias([OUTPUT_NODE])
	y = tf.matmul(y1, w2) + b2

	return y
