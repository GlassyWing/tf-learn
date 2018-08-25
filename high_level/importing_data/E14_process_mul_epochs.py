import tensorflow as tf

filenames = ["../data/file1.csv"]
filenames_placeholder = tf.placeholder(dtype=tf.string, shape=[None])
record_default = [tf.float32] * 2
dataset: tf.data.Dataset = tf.contrib.data.CsvDataset(filenames_placeholder, record_default, header=False)
dataset = dataset.batch(30).repeat(2)

print(dataset.output_shapes)

iterator = dataset.make_initializable_iterator()
next_element = iterator.get_next()

sess = tf.Session()
sess.run(iterator.initializer, feed_dict={filenames_placeholder: filenames})
count = 0
while True:

    try:
        print(sess.run(next_element))
        count += 1
    except tf.errors.OutOfRangeError:
        break
print(count) # => 10
