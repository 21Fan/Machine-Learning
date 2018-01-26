import tensorflow as tf

matrix1 = tf.constant([[3,3]]) # 衡量
matrix2 = tf.constant([[2],
                       [2]])
product = tf.matmul(matrix1,matrix2) # numpy中的矩阵相乘np.dot(m1,m2)

## method 1
sess = tf.Session()
result = sess.run(product)
print(result)
sess.close()

# method 2
with tf.Session() as sess:      #不需要关闭 
    result2 = sess.run(product)
    print(result2)


