'''
We recommend using the higher level APIs to build models when possible. Knowing TensorFlow Core is valuable for the following reasons:

Experimentation and debugging are both more straight forward when you can use low level TensorFlow operations directly.
It gives you a mental model of how things work internally when using the higher level APIs.

谷歌官方推荐使用高级api例如 keras 等
'''

'''
关于tensor的等级，张量的等级就是张量的维度
例如：
    3. # 为0维张量，一个形状为[]的标量
    [1.,2.,3.] # 为一维张量，一个形状为[3]的矢量
    [[1., 2., 3.], [4., 5., 6.]] # 为二维张量; 一个形状为[2, 3]的矩阵
    [[[1., 2., 3.]], [[7., 8., 9.]]] # 为三维张量，形状为[2, 1, 3]
    
在TensorFlow中使用numpy的数组来表示张量的值
'''

'''
0	[]	                0-D	A 0-D tensor. A scalar.
1	[D0]	            1-D	A 1-D tensor with shape [5].
2	[D0, D1]	        2-D	A 2-D tensor with shape [3, 4].
3	[D0, D1, D2]	    3-D	A 3-D tensor with shape [1, 4, 3].
n	[D0, D1, ... Dn-1]	n-D	A tensor with shape [D0, D1, ... Dn-1].
'''

'''
改变张量的形状
需要保持原有的张量的元素的数量
'''
import tensorflow as tf

rank_three_tensor = tf.ones([3, 4, 5])
matrix = tf.reshape(rank_three_tensor, [6, 10])
print(matrix.shape)
# 将张量变形到[3， 20]，此时的-1告诉了reshape方法去计算剩下的维度
matrixB = tf.reshape(matrix, [3, -1])
print(matrixB.shape)

'''
张量的格式转化
使用tf.cast
'''
float_tensor = tf.cast(tf.constant([1, 2, 3]), dtype=tf.float32)
print(float_tensor)

'''
进行张量的计算
eval()只能用于当有默认的会话运行中的情况
'''
p = tf.placeholder(tf.float32)
t = p + 1.0

'''
使用tf.get_variable()方法来创建变量是最好的方式，
此方法要求明确变量的名称，这个名称可以便于其他地方来得到这个变量，此方法也能够让你能够去获得之前创建的变量
创建变量的方式，填写名称与形状
'''
# 此时如果不配置的话就是默认使用tf.glorot_uniform_initializer来初始化变量
my_variable = tf.get_variable('my_variable', [1, 2, 3])
# 自己配置初始化方法
my_int_variable = tf.get_variable('my_int_variable', [1, 2, 3], dtype=tf.float32, initializer=tf.zeros_initializer)
# 如果使用的初始化方法是张量那么在变量声明的参数中就不应当指明形状
other_variable = tf.get_variable('other_variable', dtype=tf.float32, initializer=tf.constant([2., 3.]))

# 如何配置张量是否用于训练
# 配置变量到本地集合中，则该变量不会用于训练
my_local = tf.get_variable('my_local', shape=(), collections=[tf.GraphKeys.LOCAL_VARIABLES])
# 或者你可以直接在变量的创建中添加参数
my_non_variable = tf.get_variable('my_non_variable', shape=(), trainable=False)

# 你也可以添加自己的集合，任意的合法字符串都以用作集合名称
tf.add_to_collection('my_collection_name', my_local)
print(tf.get_collection('my_collection_name'))

'''
By default every tf.Variable gets placed in the following two collections:

tf.GraphKeys.GLOBAL_VARIABLES --- variables that can be shared across multiple devices,
tf.GraphKeys.TRAINABLE_VARIABLES --- variables for which TensorFlow will calculate gradients.
'''

# 运行设备的配置
with tf.device('/device:GPU:0'):
    v = tf.get_variable('v', [1])

# 在开始执行变量时，需要进行变量的初始化
# 在低级的api中，你需要自己清楚的进行变量的初始化，而在一些高级api中例如keras，它会自动的帮你进行初始化
tf.Session().run(tf.global_variables_initializer())
# 以上就是会初始化所有在集合tf.GraphKeys.GLOBAL_VARIABLES中的变量

# 同样，你也能够自定义想要初始化的变量
tf.Session().run(my_variable.initializer)

# 你也能够查看到还有哪些变量未被初始化
print(tf.Session().run(tf.report_uninitialized_variables()))

'''
需要注意的是，tf.global_variables_initializer并不会按照顺序初始化变量
所以在需要用到其他变量时，最好使用变量的variable.initialized_value()而不是直接用变量
'''
y = tf.get_variable('y', shape=(), initializer=tf.zeros_initializer())
w = tf.get_variable('w', initializer=y.initialized_value() + 1)
print(w)

'''
通过设置名称范围来实现变量共享
'''


# 现在我们有一层卷积层
def conv_layer(input, kernel_shape, bias_shape):
    # 创建变量名称weights
    weights = tf.get_variable('weights', kernel_shape, initializer=tf.random_normal_initializer)
    # 创建变量biases
    biases = tf.get_variable('biases', bias_shape, initializer=tf.constant_initializer(0.0))

    conv = tf.nn.conv2d(input, weights, strides=[1, 1, 1, 1], padding='SAME')

    return tf.nn.relu(conv + biases)


def my_image_filter(input_image):
    with tf.variable_scope('conv1'):
        relu1 = conv_layer(input_image, [5, 5, 32, 32], [32])
    with tf.variable_scope('conv2'):
        return conv_layer(relu1, [5, 5, 32, 32], [32])
#     此时变量就会是'conv1/weights' 'conv1/biases' 和 'conv2/weights' 'conv2/biases'

