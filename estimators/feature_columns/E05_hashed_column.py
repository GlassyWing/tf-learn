import tensorflow as tf

hashed_feature_column = tf.feature_column.categorical_column_with_hash_bucket(
    key="some_feature",
    hash_bucket_size=100)
