import tensorflow as tf

a = tf.constant(5)
b = tf.constant(6)

c = a + b

print(c)

with tf.Session() as sess:
    print(sess.run(c))