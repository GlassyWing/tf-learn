import numpy as np
import tensorflow as tf

a = tf.constant(3.0, dtype=tf.float32)
b = tf.constant(4.0)

total = a + b
print(a)
print(b)
print(total)

# writer = tf.summary.FileWriter('.')
# writer.add_graph(tf.get_default_graph())
# writer.flush()
# writer.close()

sess = tf.Session()
print(sess.run(total))
print(sess.run({'ab': (a, b), 'total': total}))
