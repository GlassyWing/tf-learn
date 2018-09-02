import tensorflow as tf

embedding_dimensions = 3

categorical_column = tf.feature_column.categorical_column_with_vocabulary_list(
    key="some_fb",
    vocabulary_list=['a', 'b', 'c'])

indicator_column = tf.feature_column.indicator_column(categorical_column)

embedding_column = tf.feature_column.embedding_column(
    categorical_column=categorical_column,
    dimension=embedding_dimensions)
