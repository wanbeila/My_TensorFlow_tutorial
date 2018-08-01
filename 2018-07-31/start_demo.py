import tensorflow as tf

### 创建图开始 ###

# initial a constant variable
# 1 x 2 创建一个矩阵 op
matrix1 = tf.contant([[3., 3.]])
# 创建另一个2 x 1矩阵 op
matrix2 = tf.constant([[2.], [4.]])

# 创建一个矩阵乘法 op
product = tf.matmul(matrix1, matrix2)

### 创建图结束 ###

### 通过创建会话session来运行图
