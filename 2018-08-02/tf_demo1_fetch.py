import tensorflow as tf
'''
通过设置传入多个tensor 来取回多个值
'''

input1 = tf.constant(1)
input2 = tf.constant(2)
input3 = tf.constant(3)

intermed = tf.add(input1, input2)
mul = tf.multiply(intermed, input3)

with tf.Session() as sess:
    result = sess.run([mul, intermed])
    print(result)
