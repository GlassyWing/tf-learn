import tensorflow as tf

filenames = ["../data/file1.csv", "../data/file2.csv"]
record_defaults = [tf.float32] * 2  # Two required float columns
dataset: tf.data.Dataset = tf.contrib.data.CsvDataset(filenames, record_defaults, header=False)

iterator = dataset.make_one_shot_iterator()
next_item = iterator.get_next()

sess = tf.Session()
while True:
    try:
        print(sess.run(next_item))
    except:
        break
