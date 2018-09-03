import tensorflow as tf

c_0 = tf.constant(0, name='c')  # => operation named "c"

# Already-used names will be "uniquified"
c_1 = tf.constant(2, name='c')

# Name scope add a prefix to all operations created in the same context.
with tf.name_scope("outer"):
    c_2 = tf.constant(2, name="c")  # => operation named "outer/c"

    # Name scopes nest like paths in a hierarchical file system.
    with tf.name_scope("inner"):
        c_3 = tf.constant(3, name="c")  # => operation named "outer/inner/c"

    # Exiting a name scope context will return to the previous prefix.\
    c_4 = tf.constant(4, name="c")  # => operation named "outer/c_1"

    # Already-used name scopes will be "uniquified".
    with tf.name_scope("inner"):
        c_5 = tf.constant(5, name="c")  # => operation named "outer/inner_1/c"
