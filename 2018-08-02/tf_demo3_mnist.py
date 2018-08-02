# MNIST数据集
import tensorflow.examples.tutorials.mnist.input_data as input_data
mnist = input_data.read_data_sets('MNIST_data/', one_hot=True)

# 实现回归模型
import tensorflow as tf

x = tf.placeholder(tf.float32, [None, 784])

w = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

y = tf.nn.softmax(tf.matmul(x, w) + b)

# 训练模型
# 为了训练模型，需要定义一定的指标来评估模型好坏
# 通常我们通过 成本（cost）或者 损失（loss）来判断，然后尽量减小损失
# 一个常见的成本函数为“交叉熵”（cross-entropy）
# 计算交叉熵
# 通过占位符来输入正确值y_
y_ = tf.placeholder('float', [None, 10])
# 然后通过公式 -∑y'log(y) 来计算
cross_entropy = -tf.reduce_sum(y_ * tf.log(y))

# 然后TensorFlow就可以通过反向传播算法（backpropagation algorithm）来有效的确定你的变量是如何影响你的成本的
# 然后TensorFlow会通过你选择的优化算法来不断地修改变量来降低成本
# 这里通过梯度下降算法以0.01的学习速率来最小化交叉熵
# 梯度下降是一种很简单的方式，即通过不断地向梯度为0的方向去改变值
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

# 开始运行前，我们需要通过添加一个操作来初始化我们的变量
init = tf.global_variables_initializer()

# 现在可以在一个会话中开始启动模型了
with tf.Session() as sess:
    # 先初始化变量
    sess.run(init)
    # 然后开始训练模型，设置训练次数1000
    for _ in range(1000):
        # 设置训练步骤中，每次随机抓取训练数据中的100个数据点，并通过这些点来替换之前的train_step
        batch_xs, batch_ys = mnist.train.next_batch(100)
        sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
        # 通过函数tf.argmax()来取出某一tensor在某一维度上的数据最大值所在的索引位置
        # (y_, 1)表示正确的标签，(y, 1)表示预测值
        # 下面的代码会得到一组布尔值，通过将其转换为浮点数，然后取平均值
        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, 'float'))
        # 计算学习到的模型在测试数据集上的正确率
        print(
            sess.run(
                accuracy,
                feed_dict={
                    x: mnist.test.images,
                    y_: mnist.test.labels
                }))
