import tensorflow as tf


def _parse_function(filename, label):
    image_string = tf.read_file(filename)
    image_decode = tf.image.decode_jpeg(image_string)
    image_resized = tf.image.resize_images(image_decode, [28, 28])
    return image_resized, label


# A vector of filenames.
filenames = tf.constant(["../data/image1.jpg", "../data/image2.jpg"])

# `labels[i]` is the label for the image in `filenames[i]`
labels = tf.constant([0, 1])

dataset = tf.data.Dataset.from_tensor_slices((filenames, labels))
dataset = dataset.map(_parse_function)


