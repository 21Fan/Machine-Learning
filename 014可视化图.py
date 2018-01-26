import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

#大框架
def add_layer(inputs,in_size,out_size,activation_function=None):
    #输入，输入大小，输出大小，激励函数类型
    with tf.name_scope('layer'):
        with tf.name_scope('weights'):
            Weights = tf.Variable(tf.random_normal([in_size,out_size]),name='W')#矩阵
        with tf.name_scope('biases'):    
            biases = tf.Variable(tf.zeros([1,out_size])+0.1)#列表
        with tf.name_scope('Wx_plus_b'):
            Wx_plus_b = tf.matmul(inputs,Weights)+biases
        if activation_function is None:
                outputs = Wx_plus_b
        else:
           outputs = activation_function(Wx_plus_b)
        return outputs

x_data = np.linspace(-1,1,300)[:,np.newaxis]
noise = np.random.normal(0,0.05,x_data.shape)
y_data = np.square(x_data) - 0.5 + noise

#placeholder放置变量
with tf.name_scope('inputs'):
    xs = tf.placeholder(tf.float32,[None,1],name='x_input')
    ys = tf.placeholder(tf.float32,[None,1],name='y_input')

l1 = add_layer(xs,1,10,activation_function=tf.nn.relu)#输入层到隐藏层
prediction = add_layer(l1,10,1,activation_function=None)#隐藏层到输出层

with tf.name_scope('loss'):
    loss = tf.reduce_mean(tf.reduce_sum((tf.square(ys - prediction)),
                     reduction_indices=[1]))
with tf.name_scope('train_step'):
    train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

init = tf.global_variables_initializer()

sess = tf.Session()
write = tf.summary.FileWriter("logs/",sess.graph)
sess.run(init)#运算开始


#plot可视化
fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.scatter(x_data,y_data)
plt.ion()
plt.show()


for i in range(1000):
    sess.run(train_step,feed_dict={xs:x_data,ys:y_data})
    if i%50 == 0:

      #  print(sess.run(loss,feed_dict={xs:x_data,ys:y_data}))
      try:
          ax.lines.remove(lines[0])
      except Exception:
          pass
      prediction_value = sess.run(prediction,feed_dict={xs:x_data})
      lines = ax.plot(x_data,prediction_value,'r-',lw=5)
      
      plt.pause(0.1)
