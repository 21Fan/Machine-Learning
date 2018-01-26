# -*- coding: utf-8 -*-
'''卷积神经网络测试MNIST数据'''
#导入MNIST数据
import tensorflow.examples.tutorials.mnist.input_data as input_data
import tensorflow as tf
import numpy as np
from PIL import Image
import cv2 as cv

mnist = input_data.read_data_sets('MNIST_data/', one_hot=True)
sess = tf.InteractiveSession()
with tf.name_scope("inputs"):
    x = tf.placeholder(tf.float32, shape=[None, 784], name="x_input")
    y_ = tf.placeholder(tf.float32, shape=[None, 10], name="y_input")


W = tf.Variable(tf.zeros([784,10]))
b = tf.Variable(tf.zeros([10]))
sess.run(tf.global_variables_initializer())

#权重初始化函数,用一个较小的正数来初始化偏置项
def weight_variable(shape):
    #tf.truncated_normal(shape,mean=0.0,stddev)：
    # shape：张量的维度；mean：正态分布的均值；stddev：正态分布的标准差
    # 从截断的正态分布中输出随机值。
    #生成的值服从具有指定平均值和标准偏差的正态分布，如果生成的值大于平均值2个标准偏差的值则丢弃重新选择。
  initial = tf.truncated_normal(shape, stddev=0.1, name='W')
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape, name='b')
  return tf.Variable(initial)

#卷积和池化函数
def conv2d(x, W):
    # strides=[1, x 方向的步长, y 方向的步长, 1]
    #padding='SAME' ：SAME 使卷积后的输出图像与原来一样， 0 填充
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')

with tf.name_scope('layer1'):
    #第一层卷积
    # 5*5 是卷积核大小， 1 是一个通道， 32 输出个数
    with tf.name_scope('weights'):
        W_conv1 = weight_variable([5, 5, 1, 32])
        tf.summary.histogram('Layer1\'s Weights', W_conv1)

    with tf.name_scope('biases'):
        b_conv1 = bias_variable([32])
        tf.summary.histogram('Layer1\'s biases', b_conv1)

    #把x变成一个4d向量
    #tf.reshape(x, [-1:先不管数据维度,28,28,颜色通道])
    x_image = tf.reshape(x, [-1,28,28,1])

    #把x_image和权值向量进行卷积，加上偏置项，然后应用ReLU激活函数，
    #输出 28*28*32
    with tf.name_scope('relu'):
        h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
    #池化，输出：14*14*32
    h_pool1 = max_pool_2x2(h_conv1)

with tf.name_scope('layer2'):
    #第二层卷积
    with tf.name_scope('weights'):
        W_conv2 = weight_variable([5, 5, 32, 64])
        tf.summary.histogram('Layer2\'s Weights', W_conv2)

    with tf.name_scope('biases'):
        b_conv2 = bias_variable([64])
        tf.summary.histogram('Layer2\'s biases', b_conv2)
    #输出 14*14*64
    with tf.name_scope('relu'):
        h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
    #输出7*7*64
    h_pool2 = max_pool_2x2(h_conv2)

#密集连接层
with tf.name_scope('full_layer'):
    with tf.name_scope('weights'):
        W_fc1 = weight_variable([7 * 7 * 64, 1024])
        tf.summary.histogram('FC\'s Weights', W_fc1)

    with tf.name_scope('biases'):
        b_fc1 = bias_variable([1024])
        tf.summary.histogram('FC\'s biases', b_fc1)

    #将第二次pooling 后的结果展成 一维 向量
    h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
    with tf.name_scope('relu'):
        h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
#为了减少过拟合，在输出层之前加入dropout
keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

with tf.name_scope('softmax'):
    #添加一个softmax层，就像softmax regression一样
    with tf.name_scope('weights'):
        W_fc2 = weight_variable([1024, 10])
        tf.summary.histogram('softmax\'s Weights', W_fc2)

    with tf.name_scope('biases'):
        b_fc2 = bias_variable([10])
        tf.summary.histogram('softmax\'s biases', b_fc2)
    y_conv=tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)


#训练设置
with tf.name_scope("loss"):
    cross_entropy = -tf.reduce_sum(y_*tf.log(y_conv)) #交叉熵
    tf.summary.scalar('loss', cross_entropy)

with tf.name_scope('train'):
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy) #ADAM优化器来做梯度最速下降

#tf.argmax(y_conv,1)返回的是模型对于任一输入x预测到的标签值，而 tf.argmax(y_,1) 代表正确的标签，
# 我们可以用 tf.equal 来检测我们的预测是否真实标签匹配(索引位置一样表示匹配)
correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
with tf.name_scope("accuracy"):
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    tf.summary.scalar('accuracy', accuracy)

sess.run(tf.global_variables_initializer())

merge = tf.summary.merge_all()
train_writer = tf.summary.FileWriter("d:/pythonCode/HNR/logs/train/", tf.get_default_graph())

#保存参数
args_save_path = 'D:/pythonCode/HNR/data_2/args.ckpt'
saver = tf.train.Saver()

#训练

# 调整内容区域位置
def adjust_content(img):
    arr = None
    if img.shape[1] != 784:
        raise Exception("不正确的数据类型！")
    for t in img:
        t.resize([28, 28])
        h = np.max(t, axis=1)
        v = np.max(t, axis=0)
        ta = np.argwhere(h)
        tb = np.argwhere(v)
        left = np.reshape(tb[0], 1)
        right = np.reshape(tb[-1:], 1)
        top = np.reshape(ta[0], 1)
        bottom = np.reshape(ta[-1:], 1)
        h_center = right-left
        cut = 28-bottom+top   # 垂直切除
        excursion_l = int(left if (cut/2) >= left else (cut/2))              # 左边偏移
        excursion_r = int((27-right) if (cut/2) >= (27-right) else (cut/2))  # 右边偏移
        # 截取关键区域图片平移
        tt = t[top[0]:bottom[0], excursion_l:27-excursion_r]
        if np.sum(tt.shape) < 6 or np.min(tt.shape) == 0:
            arr = t.reshape((1, 784)) if arr is None else np.vstack([arr, t.reshape((1, 784))])
        tt = cv.resize(tt, (28, 28), interpolation=cv.INTER_CUBIC)
        m = np.float32([[1, 0, 15-h_center], [0, 1, 0]])
        tt = cv.warpAffine(tt, m, (28, 28))
        tt = np.float32((tt > 0.2))
        tabel = (tt > 0)
        hs_max_value = np.max(np.sum(tabel, axis=0))
        vs_max_index = np.argmax(np.sum(tabel, axis=1))
        arr = tt.reshape((1, 784)) if arr is None else np.vstack([arr, tt.reshape((1, 784))])
    return arr, [hs_max_value, vs_max_index]

total_accuracy = 0
for i in range(1000):
  batch = mnist.train.next_batch(50)

  #feed_data, _ = adjust_content(batch[0])
  feed_data = batch[0]
  if i%50 == 0:
    # train_accuracy = accuracy.eval(feed_dict={
    #     x:feed_data, y_: batch[1], keep_prob: 1.0})
    train_result = sess.run(merge, feed_dict={x:feed_data, y_: batch[1], keep_prob: 1.0})
    #total_accuracy = total_accuracy + train_accuracy
    #print("-->step %d, training accuracy %.4f"%(i, train_accuracy))
    train_writer.add_summary(train_result, i)

  train_step.run(feed_dict={x: feed_data, y_: batch[1], keep_prob: 0.5})


print('\n训练完成！')
print('训练集平均识别率为：%.4f' % (total_accuracy/10))
save_path = saver.save(sess, args_save_path)
print('\n参数已经存储在：',save_path)


