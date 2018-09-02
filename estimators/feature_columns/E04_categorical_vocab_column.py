import tensorflow as tf

vocab_feature_column = tf.feature_column.categorical_column_with_vocabulary_list(
    key="",
    vocabulary_list=["kitchenware", "electronics", "sports"]
)

vocab_feature_column = tf.feature_column.categorical_column_with_vocabulary_file(
    key="",
    vocabulary_file="product_class.txt",
    vocabulary_size=3
)
