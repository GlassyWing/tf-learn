import tensorflow as tf

sess = tf.Session()
num_epochs = 100

filenames = ["../data/file1.csv", "../data/file2.csv"]
dataset: tf.data.Dataset = tf.contrib.data.CsvDataset(filenames)
dataset = dataset.shuffle(buffer_size=10000)
dataset = dataset.batch(32)
dataset = dataset.repeat(num_epochs)

iterator = dataset.make_one_shot_iterator()

next_example, next_label = iterator.get_next()
