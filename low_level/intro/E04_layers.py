import tensorflow as tf

sess = tf.Session()

x = tf.placeholder(dtype=tf.float32, shape=[None, 3])
linear_model = tf.layers.Dense(units=1)
y = linear_model(x)

init = tf.global_variables_initializer()
sess.run(init)

writer = tf.summary.FileWriter('.')
writer.add_graph(tf.get_default_graph())
writer.flush()

print(sess.run(y, {x: [[1, 2, 3], [4, 5, 6]]}))

