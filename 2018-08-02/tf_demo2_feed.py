import tensorflow as tf
'''
TensorFlow提供了feed机制，该机制可以临时代替图中的任意操作中的tensor
'''
input1 = tf.placeholder(tf.float32)
input2 = tf.placeholder(tf.float32)

mul = tf.multiply(input1, input2)

with tf.Session() as sess:
    result = sess.run([mul], feed_dict={input1: [7.], input2: [8.]})
    print(result)