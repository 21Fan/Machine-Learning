import tensorflow as tf

state = tf.Variable(0,name='counter')
#print(state.name)
one = tf.constant(1)

new_value = tf.add(state,one)
update = tf.assign(state,new_value)

init = tf.global_variables_initializer() # 如果有定义变量一定要用

with tf.Session() as sess:
    sess.run(init) # 必须初始化
    for _ in range(3):
        sess.run(update)                                            
        print(sess.run(state))
