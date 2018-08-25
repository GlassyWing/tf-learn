import numpy as np
import tensorflow as tf

##### 1. create .npz file #####
# features = np.random.rand(20, 2)
# labels = np.random.rand(20, 1) > 0.5
#
# print(features)
# print(labels)
#
# np.savez("training.npz", features=features, labels=labels)

##### 2. load to memory #####
# with np.load("../data/training.npz") as data:
#     features = data["features"]
#     labels = data["labels"]
#
# # Assume that each row of `features` corresponds to the same row as `labels`.
# assert features.shape[0] == labels.shape[0]
#
# dataset = tf.data.Dataset.from_tensor_slices((features, labels))

##### 3. load to memory #####
# Load the training data into two NumPy arrays, for example using `np.load()`.
with np.load("/var/data/training_data.npy") as data:
    features = data["features"]
    labels = data["labels"]

# Assume that each row of `features` corresponds to the same row as `labels`.
assert features.shape[0] == labels.shape[0]

features_placeholder = tf.placeholder(features.dtype, features.shape)
labels_placeholder = tf.placeholder(labels.dtype, labels.shape)

dataset = tf.data.Dataset.from_tensor_slices((features_placeholder, labels_placeholder))
# [Other transformations on `dataset`...]
# dataset = ...
iterator = dataset.make_initializable_iterator()

sess = tf.Session()
sess.run(iterator.initializer, feed_dict={features_placeholder: features, labels_placeholder: labels})
