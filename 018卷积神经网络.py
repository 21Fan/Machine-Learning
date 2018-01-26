import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data',one_hot=True)

#对比正确的
def compute_accuracy(v_xs,v_ys):
    global prediction
    #预测值:概率
    y_pre = sess.run(prediction,feed_dict={xs:v_xs,keep_prob:1})
    #tf.argmax(vector, 1)返回的是vector中的最大值的下标
    #如果vector是一个向量，那就返回一个值，如果是一个矩阵，那就返回一个向量
    #tf.equal(A, B)对比向量各项相等与否
    correct_prediction = tf.equal(tf.argmax(y_pre,1),tf.argmax(v_ys,1))
    #tf.reduce_mean求平均
    #tf.cast(x,dtype,name)类型转换
    accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
    result = sess.run(accuracy,feed_dict={xs:v_xs,ys:v_ys,keep_prob:1})
    return result

def weight_variable(shape):
    initial = tf.truncated_normal(shape,stddev=0.1)
    return tf.Variable(initial)

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
keep_prob = tf.placeholder(tf.float32)

x_image = tf.reshape(xs,[-1,28,28,1])
#print(x_image.shape) 

##conv1 layer##
# patch 5x5,in size 1是image的厚度,out size 32高度
W_conv1 = weight_variable([5,5,1,32])
b_conv1 = bias_variable([32])
h_conv1 = tf.nn.relu(conv2d(x_image,W_conv1)+b_conv1)#out size=28x28x32
h_pool1 = max_pool_2x2(h_conv1)                     #out size=14*14*32

##conv2 layer##
W_conv2 = weight_variable([5,5,32,64])
# patch 5x5,in size 32输入厚度,out size 64高度
b_conv2 = bias_variable([64])
h_conv2 = tf.nn.relu(conv2d(h_pool1,W_conv2)+b_conv2)#out size=14*14*64
h_pool2 = max_pool_2x2(h_conv2)                     #out size=7*7*64

##func1 layer##
W_fc1 = weight_variable([7*7*64,1024])
b_fc1 = bias_variable([1024])
#[n_samples,7,7,64]->>[n_samples,7*7*64]
h_pool2_flat = tf.reshape(h_pool2,[-1,7*7*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat,W_fc1)+b_fc1)
h_fc1_drop = tf.nn.dropout(h_fc1,keep_prob)

##func1 layer##
W_fc2 = weight_variable([1024,10])
b_fc2 = bias_variable([10])
prediction = tf.nn.softmax(tf.matmul(h_fc1_drop,W_fc2)+b_fc2)

#分类算法
cross_entropy = tf.reduce_mean(-tf.reduce_sum(ys * tf.log(prediction),
                     reduction_indices=[1])) # loss
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

sess = tf.Session()
#重要步骤
sess.run(tf.global_variables_initializer())

for i in range(1000):
    #随机梯度下降
    batch_xs,batch_ys = mnist.train.next_batch(100)
    
    sess.run(train_step,feed_dict={xs:batch_xs,ys:batch_ys})
    if i % 50 == 0:
        print(compute_accuracy(mnist.test.images,mnist.test.labels))
        #train data和test data分开




