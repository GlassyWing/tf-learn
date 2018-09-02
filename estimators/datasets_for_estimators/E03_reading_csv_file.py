import tensorflow as tf

import estimators.premade_estimators.iris_data as iris_data

train_path, test_path = iris_data.maybe_download()

ds = tf.data.TextLineDataset(train_path).skip(1)

ds = ds.map(iris_data._parse_line)

print(ds)