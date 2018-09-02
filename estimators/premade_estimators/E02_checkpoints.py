import tensorflow as tf

import estimators.premade_estimators.iris_data as iris_data

batch_size = 100

(train_x, train_y), (test_x, test_y) = iris_data.load_data()

my_feature_columns = [tf.feature_column.numeric_column(x) for x in train_x.keys()]

classifier = tf.estimator.DNNClassifier(
    feature_columns=my_feature_columns,
    hidden_units=[10, 10],
    n_classes=3,
    model_dir='../data/models/iris')

classifier.train(input_fn=lambda: iris_data.train_input_fn(train_x, train_y, batch_size), steps=200)
