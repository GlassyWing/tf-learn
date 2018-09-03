import tensorflow as tf


def conv_relu(input, kernel_shape, bias_shape):
    # Create variable named "weights".
    weights = tf.get_variable("weights", kernel_shape,
                              initializer=tf.random_normal_initializer())
    # Create variable named "biases"
    biases = tf.get_variable("biases", bias_shape,
                             initializer=tf.zeros_initializer())

    conv = tf.nn.conv2d(input, weights, strides=[1, 1, 1, 1], padding="SAME")
    return tf.nn.relu(conv + biases)


input1 = tf.random_normal([1, 10, 10, 32])
input2 = tf.random_normal([1, 20, 20, 32])


# x = conv_relu(input1, kernel_shape=[5, 5, 32, 32], bias_shape=[32])
#  This fails. Repeat to create param: 'weights','biases'
# x = conv_relu(x, kernel_shape=[5, 5, 32, 32], bias_shape=[32])


def my_image_filter(input_images):
    with tf.variable_scope('conv1'):
        # Variables created here will be named "conv1/weights", "conv1/biases".
        relu1 = conv_relu(input_images, [5, 5, 32, 32], [32])
    with tf.variable_scope('conv2'):
        # Variables created here will be named "conv2/weights", "conv2/biases".
        return conv_relu(relu1, [5, 5, 32, 32], [32])


# If you do want the variables to be shared, you have two options. First, you can create a scope with the same name using reuse=True:
with tf.variable_scope("model"):
    output1 = my_image_filter(input1)
with tf.variable_scope("model", reuse=True):
    output2 = my_image_filter(input2)

# You can also call scope.reuse_variables() to trigger a reuse:
# with tf.variable_scope("model") as scope:
#     output1 = my_image_filter(input1)
#     scope.reuse_variables()
#     output2 = my_image_filter(input2)
