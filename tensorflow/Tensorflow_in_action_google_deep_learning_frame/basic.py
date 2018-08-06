#coding=utf-8
import tensorflow as tf

matrix1=tf.constant([[3.,3.]])
matrix2=tf.constant([[2.],[2.]])
product=tf.matmul(matrix1,matrix2)
'''
sess=tf.Session()
sess.run(tf.global_variables_initializer())
print(sess.run(product))

sess.close()
'''
with tf.Session() as sess:
    with tf.device("/gpu:0"):
        print(sess.run(product))
