import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data',one_hot=True)

#对比正确的
def compute_accuracy(v_xs,v_ys):
    global prediction
    #预测值:概率
    y_pre = sess.run(prediction,feed_dict={xs:v_xs})
    #tf.argmax(vector, 1)返回的是vector中的最大值的下标
    #如果vector是一个向量，那就返回一个值，如果是一个矩阵，那就返回一个向量
    #tf.equal(A, B)对比向量各项相等与否
    correct_prediction = tf.equal(tf.argmax(y_pre,1),tf.argmax(v_ys,1))
    #tf.reduce_mean求平均
    #tf.cast(x,dtype,name)类型转换
    accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
    result = sess.run(accuracy,feed_dict={xs:v_xs,ys:v_ys})
    return result

def weight_variable(shape):
    initial = tf.truncated(shape,stddev=0.1)
    return tf.Veriable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1,shape=shape)
    return tf.Variable(initial)

def conv2d(x,W):
    #stride[1,x的移动，y的移动,1]
    return tf.nn.conv2d(x,W,strides=[1,1,1,1],padding='SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')

xs = tf.placeholder(tf.float32,[None,784]) # 28*28
ys = tf.placeholder(tf.float32,[None,10])
keep_prob = tf.placeholder(tf.float)




