a = ['a', 'b', 'c']
b = [1, 2, 3]

features = dict(zip(a, b))

print(features.pop('c'))
print(features)

import tensorflow as tf

print(tf.newaxis)