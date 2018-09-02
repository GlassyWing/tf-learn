import tensorflow as tf

numeric_feature_column = tf.feature_column.numeric_column(key="Year")

bucketized_feature_column = tf.feature_column.bucketized_column(
    source_column=numeric_feature_column,
    boundaries=[1960, 1980, 2000])


