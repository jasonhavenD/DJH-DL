import tensorflow as tf

state=tf.Variable(0,name='counter')

one=tf.constant(1)
next=tf.add(state,one)
update=tf.assign(state,next)

init_op = tf.initialize_all_variables()
with tf.Session() as sess:
    # 运行 'init' op
    sess.run(init_op)
    # 运行 op, 更新 'state', 并打印 'state'
    for _ in range(3):
        sess.run(update)
        print(sess.run(state))

