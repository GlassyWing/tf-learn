import tensorflow as tf

sess = tf.Session()

v: tf.Variable = tf.get_variable("v", shape=(), initializer=tf.zeros_initializer())
assignment = v.assign_add(1)
sess.run(tf.global_variables_initializer())
print(sess.run(assignment))

with tf.control_dependencies([assignment]):
    w = v.read_value()  # w is guaranteed to reflect v's value after the assign_add operation.
