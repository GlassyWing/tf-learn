import tensorflow as tf

sess = tf.Session()

filenames = ["../data/file1.csv", "../data/file2.csv"]
filenames_placeholder = tf.placeholder(tf.string, shape=[None])
record_default = [tf.float32] * 2
dataset: tf.data.Dataset = tf.contrib.data.CsvDataset(filenames_placeholder, record_default)

dataset.shuffle(buffer_size=10000)
dataset = dataset.batch(32)
