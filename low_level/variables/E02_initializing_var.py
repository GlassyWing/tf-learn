import tensorflow as tf

sess = tf.Session()

# Initializing all vars in tf.GraphKeys.GLOBAL_VARIABLES
sess.run(tf.global_variables_initializer())
# Now all variables are initialized.

uninitalized_var = tf.get_variable("uninitialized_var", shape=())

print(sess.run(tf.report_uninitialized_variables()))

# Dependencies
v = tf.get_variable("v", shape=(), initializer=tf.zeros_initializer())
w = tf.get_variable("w", initializer=v.initialized_value() + 1)
