import tensorflow as tf

vec = tf.random_uniform(shape=(3,))
out1 = vec + 1
out2 = vec + 2

sess = tf.Session()
print(sess.run(vec))
print(sess.run(vec))
print(sess.run((out1, out2)))

writer = tf.summary.FileWriter('.')
writer.add_graph(tf.get_default_graph())
writer.flush()
writer.close()