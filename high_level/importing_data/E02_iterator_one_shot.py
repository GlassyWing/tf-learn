import tensorflow as tf

sess = tf.Session()

dataset = tf.data.Dataset.range(100)
print(dataset.output_shapes)
print(dataset.output_types)

iterator = dataset.make_one_shot_iterator()
next_element = iterator.get_next()

for i in range(100):
    value = sess.run(next_element)
    assert i == value
