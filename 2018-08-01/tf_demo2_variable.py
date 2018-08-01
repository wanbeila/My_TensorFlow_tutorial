import tensorflow as tf

# 创建一个变量，初始化为标量0
state = tf.Variable(0, name='counter')

# 创建一个op，其作用是让state加1
one = tf.constant(1)

new_value = tf.add(state, one)
update = tf.assign(state, new_value)

# 启动图后，变量则必须经过'初始化'（init）op初始化
# 必须增加一个初始化op到图中

init_op = tf.global_variables_initializer()

# init session ,run graph ,run op
with tf.Session() as sess:
    sess.run(init_op)
    # 打印state的初始值
    print(sess.run(state))
    # run the op to update state,and print
    sess.run(update)
    print(sess.run(state))
