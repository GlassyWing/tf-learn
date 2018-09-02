import tensorflow as tf
import estimators.premade_estimators.iris_data as iris_data


def train_input_fn(features, labels, batch_size):
    # Convert the inputs to a Dataset.
    dataset = tf.data.Dataset.from_tensor_slices((dict(features), labels))

    # Shuffle, repeat, and batch the examples.
    dataset = dataset.shuffle(1000).repeat().batch(batch_size)

    # Return the dataset.
    return dataset


(train_x, train_y), (test_x, test_y) = iris_data.load_data()

dataset = tf.data.Dataset.from_tensor_slices((dict(train_x), train_y))
print(dataset)
print(dataset.output_shapes)
