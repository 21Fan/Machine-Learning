import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data',one_hot=True)

#架构
def add_layer(inputs,in_size,out_size,activation_function=None):#不定义激励
    Weights = tf.Variable(tf.random_normal([in_size,out_size]))#矩阵
    biases = tf.Variable(tf.zeros([1,out_size])+0.1)#列表
    Wx_plus_b = tf.matmul(inputs,Weights)+biases
    if activation_function is None:
        outputs = Wx_plus_b
    else:
        outputs = activation_function(Wx_plus_b)
    return outputs

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


xs = tf.placeholder(tf.float32,[None,784]) # 28*28
ys = tf.placeholder(tf.float32,[None,10])

#一般softmax用于classfication
prediction = add_layer(xs,784,10,activation_function=tf.nn.softmax)

#分类算法
cross_entropy = tf.reduce_mean(-tf.reduce_sum(ys * tf.log(prediction),
                     reduction_indices=[1])) # loss

train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

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
 
