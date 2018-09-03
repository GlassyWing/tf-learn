import tensorflow as tf

# Build your graph.
x = tf.constant([[37.0, -23.0], [1.0, 4.0]])
w = tf.Variable(tf.random_uniform([2, 2]))
y = tf.matmul(x, w)

out = tf.nn.softmax(y)
label = tf.random_uniform([2, 2])

loss = tf.losses.mean_squared_error(labels=label, predictions=out)
train_op = tf.train.AdadeltaOptimizer(0.01).minimize(loss)


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    # `sess.graph` provides access to the graph used in a tf.Session
    writer = tf.summary.FileWriter("../../data/logs/", sess.graph)

    # Perform your computation...
    for i in range(1000):
        sess.run(train_op)

    writer.close()
