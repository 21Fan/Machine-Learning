import tensorflow as tf
import numpy as np
#只能保存variable，不能保留框架

W = tf.Variable(np.arange(6).reshape((2,3)),dtype=tf.float32,name="weights")
b = tf.Variable(np.arange(3).reshape((1,3)),dtype=tf.float32,name="biases")

# 不需要initial
saver = tf.train.Saver()
with tf.Session()as sess:
    saver.restore(sess,"my_net/save_net.ckpt")
    print("weights:",sess.run(W))
    print("biases:",sess.run(b))
