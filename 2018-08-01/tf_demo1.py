import tensorflow as tf

# 创建一个常量 op，产生一个 1x2 矩阵，这个 op 被作为一个节点
# 加到默认图中
#
# 构造器的返回值代表该常量 op 的返回值
matrix1 = tf.constant([[3., 3.]])

# 创建另一个常量 op
matrix2 = tf.constant([[2.], [2.]])

# 创建一个乘法的 op，把以上两个常量op作为输入
product = tf.matmul(matrix1, matrix2)

# 创建会话来启动图
# sess = tf.Session()
# '''
# 调用sess的run方法来执行矩阵乘法，传入参数为'product'，
# 'product'代表了矩阵乘法op的输出，传入它是向方法表明，我们希望取回矩阵乘法op的输出
# '''
# result = sess.run(product)
# print(result)

# sess.close()
'''
也可以通过with方法来创建会话
"/cpu:0": 机器的 CPU.
"/gpu:0": 机器的第一个 GPU, 如果有的话.
"/gpu:1": 机器的第二个 GPU, 以此类推.
'''
with tf.Session() as sess:
    # 配置TensorFlow所调用的硬件
    with tf.device('/cpu:1'):
        result = sess.run(product)
        print(result)
