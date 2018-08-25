import tensorflow as tf

filenames = ["../data/file1.txt", "../data/file2.txt"]
filenames_placeholder = tf.placeholder(tf.string, [None])
dataset = tf.data.TextLineDataset(filenames_placeholder)

dataset = dataset.filter(lambda line: tf.not_equal(tf.substr(line, 0, 1), "#"))

iterator = dataset.make_initializable_iterator()
next_item = iterator.get_next()

sess = tf.Session()
sess.run(iterator.initializer, feed_dict={filenames_placeholder: filenames})

while True:
    try:
        print(sess.run(next_item))
    except tf.errors.OutOfRangeError:
        break
