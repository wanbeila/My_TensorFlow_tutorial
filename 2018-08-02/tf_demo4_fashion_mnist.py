import tensorflow as tf
from tensorflow import keras

# 帮助库
import numpy as np
import matplotlib.pyplot as plt

# 我们会使用60000图像数据来训练模型，并使用10000数据来测试
# import and load the data
fashion_mnist = keras.datasets.fashion_mnist

# 分别为训练数据和测试数据，其中图像为28 x 28的numpy数组，
# 像素值为0-255之间；标签为0-9的数字，分别对应衣服的不同类型
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

# each image is mapped to a single label
# store the labels here to use later
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
# 接下来查看一下训练的数据的一些特性
# print(train_images.shape, len(train_labels), train_labels, test_images.shape)

# use the matplotlib to show the image
plt.figure()
plt.imshow(train_images[0])
plt.colorbar()
plt.gca().grid(False)
plt.show()

# 需要将图片的像素变为0-1的浮点数，则同时将训练数据与测试数据进行/255处理
train_images = train_images / 255.0
test_images = test_images / 255.0

# 显示前25张图片，并且下面附上标签名称
plt.figure(figsize=(10, 10))
for i in range(25):
    plt.subplot(5, 5, i + 1)
    plt.xticks([])
    plt.yticks([])
    plt.grid('on')
    plt.imshow(train_images[i], cmap=plt.cm.binary)
    plt.xlabel(class_names[train_labels[i]])
plt.show()

# 神经网络的最基本的块就是神经层layer
# 神经层通过提取数据中的特征部分来使得数据变得有意义

# 神经网络的第一层将数据从2维格式化到1维，比如将28 x 28格式化为784
model = keras.Sequential([
    # 这一层神经层只进行数据的格式化并不进行学习
    keras.layers.Flatten(input_shape=(28, 28)),
    # 紧接着为两层密集连接层或者全连接层
    # 第一个层包括有128个节点，第二个层包括10个节点的softmax层，用于返回十个不同的概率值
    keras.layers.Dense(256, activation=tf.nn.relu),
    keras.layers.Dense(10, activation=tf.nn.softmax)
])

# 在进行编译之前，还需要一点设置
# 需要配置loss函数，用于反应模型训练的精确度，以便于训练朝着正确的方向前进
# 配置optimizer函数，通过设置的函数来减小loss
# 配置metrics函数，用于度量训练的程度，即精确度accuracy
model.compile(optimizer=tf.train.AdamOptimizer(),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# 训练模型，包括：
# 向模型添加训练数据
# 模型学着去关联图像和标签
# 我们要求模型去预测测试集

# 开始训练，通过调用fit()方法
model.fit(train_images, train_labels, epochs=5)

# 先在测试模型在测试集上的精确度
test_loss, test_acc = model.evaluate(test_images, test_labels)
# print('test accuracy:', test_acc)


# 做个预测
predictions = model.predict(test_images)

print(np.argmax(predictions[0]), test_labels[0])

# 接下来我们将展示一些图像及其预测值
# 先展示25张测试图像
plt.figure(figsize=(10, 10))
for _ in range(25):
    plt.subplot(5, 5, _ + 1)
    plt.xticks([])
    plt.yticks([])
    plt.grid('off')
    plt.imshow(test_images[_], cmap=plt.cm.binary)
    predicted_label = np.argmax(predictions[_])
    true_label = test_labels[_]
    if predicted_label == true_label:
        color = 'green'
    else:
        color = 'red'
    plt.xlabel('{} ({})'.format(class_names[predicted_label], class_names[true_label]), color=color)
plt.show()


# 接下来使用模型来预测一张图片
img = test_images[0]
print('single image shape ', img.shape)
# keras模型在预测时需要输入集合，因此我们需要将单张图片添加到列表中
img = (np.expand_dims(img, 0))
print('after expand image shape', img.shape)

# 现在来预测这单张的图片内容
prediction = model.predict(img)
print(class_names[np.argmax(prediction[0])])

