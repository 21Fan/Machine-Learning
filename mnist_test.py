# -*- coding: utf-8 -*-
'''卷积神经网络测试MNIST数据'''
#导入MNIST数据
import tensorflow.examples.tutorials.mnist.input_data as input_data
import tensorflow as tf
import numpy as np
from PIL import Image
import os
import cv2 as cv

mnist = input_data.read_data_sets('MNIST_data/', one_hot=True)
sess = tf.InteractiveSession()
x = tf.placeholder(tf.float32, shape=[None, 784])
y_ = tf.placeholder(tf.float32, shape=[None, 10])
W = tf.Variable(tf.zeros([784,10]))
b = tf.Variable(tf.zeros([10]))

#权重初始化函数,用一个较小的正数来初始化偏置项
def weight_variable(shape):
    #tf.truncated_normal(shape,mean=0.0,stddev)：
    # shape：张量的维度；mean：正态分布的均值；stddev：正态分布的标准差
    # 从截断的正态分布中输出随机值。
    #生成的值服从具有指定平均值和标准偏差的正态分布，如果生成的值大于平均值2个标准偏差的值则丢弃重新选择。
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)
def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

#卷积和池化函数
def conv2d(x, W):
    # strides=[1, x 方向的步长, y 方向的步长, 1]
    #padding='SAME' ：SAME 使卷积后的输出图像与原来一样， 0 填充
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')

#第一层卷积
# 5*5 是卷积核大小， 1 是一个通道， 32 输出个数
W_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])

#把x变成一个4d向量
#tf.reshape(x, [-1:先不管数据维度,28,28,颜色通道])
x_image = tf.reshape(x, [-1,28,28,1])

#把x_image和权值向量进行卷积，加上偏置项，然后应用ReLU激活函数，
#输出 28*28*32
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
#池化，输出：14*14*32
h_pool1 = max_pool_2x2(h_conv1)

#第二层卷积
W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])
#输出 14*14*64
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
#输出7*7*64
h_pool2 = max_pool_2x2(h_conv2)

#密集连接层
W_fc1 = weight_variable([7 * 7 * 64, 1024])
b_fc1 = bias_variable([1024])
#将第二次pooling 后的结果展成 一维 向量
h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
#为了减少过拟合，在输出层之前加入dropout
keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

#添加一个softmax层，就像softmax regression一样
W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])
y_conv=tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)
#训练设置
cross_entropy = -tf.reduce_sum(y_*tf.log(y_conv)) #交叉熵
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy) #ADAM优化器来做梯度最速下降

#tf.argmax(y_conv,1)返回的是模型对于任一输入x预测到的标签值，而 tf.argmax(y_,1) 代表正确的标签，
# 我们可以用 tf.equal 来检测我们的预测是否真实标签匹配(索引位置一样表示匹配)
correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

sess.run(tf.global_variables_initializer())

"""
    以下为参数恢复
"""
args_save_path = 'D:/pythonCode/HNR/data_2/args.ckpt'
# args_save_path = 'D:\\pythonCode\\HNR\\data\\handwrite'
saver = tf.train.Saver()
saver.restore(sess, args_save_path)
print('\n恢复参数成功！\n')

##########################
# 以下是从mnist中取出测试样本的过程，在实验识别图片时可注释，以节约资源
#最终评估
# print('测试集准确度：\n')
# #切片测试，将分成 batch_num 次放入， 每次放入 batch_size
# batch_size = 50
# batch_num = int(mnist.test.num_examples / batch_size)
# test_accuracy = 0
#
# for i in range(batch_num):
#     batch = mnist.test.next_batch(batch_size)
#     test_accuracy += accuracy.eval(feed_dict={x: batch[0],
#                                               y_: batch[1],
#                                               keep_prob: 1.0})
#
# test_accuracy /= batch_num
# print("test accuracy %g\n" % test_accuracy)
############################



######################
#   以下为测试实际图片代码

# 调整内容区域位置
def adjust_content(img, name):
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

        # # 保存调整后的图片
        # image_1 = tt * 255
        # image_2 = Image.fromarray(image_1)
        # if image_2.mode != 'RGB':
        #     image_2 = image_2.convert('RGB')
        # path = 'e:/imags/adjust/' + name
        # image_2.save(path)

    return arr, [hs_max_value, vs_max_index]

count = 0   #记录图片总数以计算正确率
right = 0   #记录正确识别个数
toprint = 100  #作用是只打印出一次，以供查看
test_path = "E:/Imags/myimag_temp/"  #需要识别的自定义的图片所在的文件夹

for root, dirs, files in os.walk(test_path):
    for name in files:
        img_path = root + name

        imm = np.array(Image.open(img_path).convert('L'))
        imm_1 = imm/255
        imm_2 = Image.fromarray(imm_1)  # 转化为图像
        imm_3 = imm_2.resize([28, 28])  # 压缩
        im_array = np.array(imm_3)  # 转化为数组

        imm_4 = im_array.reshape((1, 784))  # 转化为符合验证一维的数组

        #imm_4,_ = adjust_content(imm_4, name)
        # toprint -= 1
        # if toprint == 1:
        #     print(imm_4.reshape((28,28)))
        #     print(test_x.reshape((28,28)))
        #     a_1 = imm_4 * 255
        #     a_2 = test_x * 255
        #     show_1 = Image.fromarray(a_1.reshape((28,28)))
        #     show_2 = Image.fromarray(a_2.reshape((28,28)))
        #     show_1.show()
        #     show_2.show()
            # toprint = False


        a=[[1,0,0,0,0,0,0,0,0,0],[0,1,0,0,0,0,0,0,0,0],[0,0,1,0,0,0,0,0,0,0],[0,0,0,1,0,0,0,0,0,0]
           ,[0,0,0,0,1,0,0,0,0,0],[0,0,0,0,0,1,0,0,0,0],[0,0,0,0,0,0,1,0,0,0],[0,0,0,0,0,0,0,1,0,0]
           ,[0,0,0,0,0,0,0,0,1,0],[0,0,0,0,0,0,0,0,0,1]]
        count = count + 1

        '''
        乐佳计算准确率的方法
        '''
        label = np.zeros([1, 10])
        label[:, int(name[1])] = 1
        pred = sess.run(y_conv, feed_dict={x: imm_4, keep_prob: 1.0})
        check = (np.argmax(label, 1) == np.argmax(pred, 1))
        if check:
            right = right + 1
            print('正确识别出：', name)
            print('识别数字为：', np.argmax(pred, 1))

        '''
        原版计算准确率的方法
        
        # for i in range(len(a)):
        #     test=accuracy.eval(feed_dict={x: imm_4, y_: [a[i]], keep_prob: 1.0})
        #     if test>0.9:
        #         #print('识别数字为：',i)
        #         if i == int(name[1]):
        #             right = right + 1
        #             print('正确识别出：', name)
        #             print('识别数字为：', i)
        #         #     imags = Image.fromarray(imm)
        #         #     imags.save('E:/Imags/right/'+name)
        #         # else:
        #         #     imags = Image.fromarray(imm)
        #         #     imags.save('E:/Imags/error/' + name)
        #         break
        '''

print('\n识别正确数：',right)
print('总数：',count)
print('自定义图片的识别率：%.4f' % ((right/count)*100))
