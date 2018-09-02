import tensorflow as tf

sess = tf.Session()

my_variable: tf.Variable = tf.get_variable("my_variable", [1, 2, 3])
my_int_var = tf.get_variable("my_int_variable", [1, 2, 3], initializer=tf.zeros_initializer)
other_var = tf.get_variable("other_variable", initializer=tf.constant([23, 42]))

init = tf.global_variables_initializer()
sess.run(init)

print(sess.run(my_variable))
print(sess.run(my_int_var))
print(sess.run(other_var))

my_local = tf.get_variable('my_local', shape=(), collections=[tf.GraphKeys.LOCAL_VARIABLES])
my_non_trainable = tf.get_variable('my_non_trainable', shape=(), trainable=False)

# Add own collection

tf.add_to_collection('my_collection_name', my_local)

print(tf.get_collection('my_collection_name'))
