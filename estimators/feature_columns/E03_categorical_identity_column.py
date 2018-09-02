import tensorflow as tf

identity_feature_column = tf.feature_column.categorical_column_with_identity(
    key="my_feature_b",
    num_buckets=4)
