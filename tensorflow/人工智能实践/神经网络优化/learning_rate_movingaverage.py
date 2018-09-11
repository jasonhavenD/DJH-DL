#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name:learning_rate_movingaverage
   Author:jasonhaven
   date:18-9-10
-------------------------------------------------
   Change Activity:18-9-10:
-------------------------------------------------
"""
import tensorflow as tf

# 定义变量
# 定义一个32位浮点变量，初始值为0.0  这个代码就是不断更新w1参数，优化w1参数，滑动平均做了个w1的影子
w1 = tf.Variable(0, dtype=tf.float32)

# 定义num_updates（NN的迭代轮数）,初始值为0，不可被优化（训练），这个参数不训练
global_step = tf.Variable(0, trainable=False)

MOVING_AVERAGE_DECAY = 0.99

ema = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)

# ema.apply后的括号里是更新列表，每次运行sess.run（ema_op）时，对更新列表中的元素求滑动平均值。
# 在实际应用中会使用tf.trainable_variables()自动将所有待训练的参数汇总为列表
ema_op = ema.apply(tf.trainable_variables())

# 2. 查看不同迭代中变量取值的变化。
with tf.Session() as sess:
	init_op = tf.global_variables_initializer()

	sess.run(init_op)

	print('当前 global_step ', sess.run(global_step))
	# 用ema.average(w1)获取w1滑动平均值 （要运行多个节点，作为列表中的元素列出，写在sess.run中）
	print("current w1", sess.run([w1, ema.average(w1)]))

	# 更新global_step和w1的值, 模拟出轮数为100时，参数w1变为10, 以下代码global_step保持为100，每次执行滑动平均操作，影子值会更新
	sess.run(tf.assign(global_step, 100))
	sess.run(tf.assign(w1, 10))

	# 每次sess.run会更新一次w1的滑动平均值
	sess.run(ema_op)
	print('当前 global_step ', sess.run(global_step))
	# 用ema.average(w1)获取w1滑动平均值 （要运行多个节点，作为列表中的元素列出，写在sess.run中）
	print("current w1", sess.run([w1, ema.average(w1)]))

	sess.run(ema_op)
	print('当前 global_step ', sess.run(global_step))
	# 用ema.average(w1)获取w1滑动平均值 （要运行多个节点，作为列表中的元素列出，写在sess.run中）
	print("current w1", sess.run([w1, ema.average(w1)]))

	sess.run(ema_op)
	print('当前 global_step ', sess.run(global_step))
	# 用ema.average(w1)获取w1滑动平均值 （要运行多个节点，作为列表中的元素列出，写在sess.run中）
	print("current w1", sess.run([w1, ema.average(w1)]))

	sess.run(ema_op)
	print('当前 global_step ', sess.run(global_step))
	# 用ema.average(w1)获取w1滑动平均值 （要运行多个节点，作为列表中的元素列出，写在sess.run中）
	print("current w1", sess.run([w1, ema.average(w1)]))

	sess.run(ema_op)
	print('当前 global_step ', sess.run(global_step))
	# 用ema.average(w1)获取w1滑动平均值 （要运行多个节点，作为列表中的元素列出，写在sess.run中）
	print("current w1", sess.run([w1, ema.average(w1)]))
