#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name:app
   Author:jasonhaven
   date:18-9-11
-------------------------------------------------
   Change Activity:18-9-11:
-------------------------------------------------
"""
import os
from PIL import Image
import numpy as np
import tensorflow as tf
import mnist_forward
import mnist_backward


def prepocess(pic_path):
	# 读入图片
	img = Image.open(pic_path)
	# 缩放大小，设定ANTIALIAS，即抗锯齿
	reIm = img.resize((28, 28), Image.ANTIALIAS)
	#
	im_arr = np.array(img.convert('L'))  # PIL的九种不同模式：1，L，P，RGB，RGBA，CMYK，YCbCr,I，F

	# 给图片做二值化处理（0,255），让图片只有白色和黑色，过滤噪声
	threshold = 50
	for i in range(im_arr.shape[0]):
		for j in range(im_arr.shape[1]):
			im_arr[i][j] = 255 - im_arr[i][j]
			if (im_arr[i][j] < threshold):
				im_arr[i][j] = 0
			else:
				im_arr[i][j] = 255
	nm_arr = im_arr.reshape([1, 784])
	nm_arr = nm_arr.astype(np.float32)
	img_ready = np.multiply(nm_arr, 1.0 / 255.0)

	return img_ready


def predict_with_model(testPicArr):
	with tf.Graph().as_default() as g:
		x = tf.placeholder(tf.float32, [None, mnist_forward.INPUT_NODE])
		y = mnist_forward.forward(x, None)
		preValue = tf.argmax(y, 1)

		ema = tf.train.ExponentialMovingAverage(mnist_backward.MOVING_AVERAGE_DECAY)
		ema_restore = ema.variables_to_restore()
		saver = tf.train.Saver(ema_restore)

		with tf.Session() as sess:
			ckpt = tf.train.get_checkpoint_state(mnist_backward.MODEL_SAVE_PATH)
			if ckpt and ckpt.model_checkpoint_path:
				saver.restore(sess, ckpt.model_checkpoint_path)
				preValue = sess.run(preValue, feed_dict={x: testPicArr})
				return preValue
			else:
				print("No checkpoint file found")
				return -1


def application():
	print('Input the number of test pictures:')
	test_num = int(input())
	for i in range(test_num):
		pic_path = os.path.join(os.getcwd(), 'picture', str(i) + '.png')
		x = prepocess(pic_path)
	y_pred = predict_with_model(x)
	print("picture %s is predicted as %d" % (str(i) + '.png', y_pred))


if __name__ == '__main__':
	application()
