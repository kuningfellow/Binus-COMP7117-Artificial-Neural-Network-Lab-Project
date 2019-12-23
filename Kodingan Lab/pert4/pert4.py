# pip install tensorflow==1.10
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, OrdinalEncoder
import numpy as np
from sklearn.model_selection import train_test_split

# Ngambil data dari csv
def load_dataset():
    data = pd.read_csv('Iris-modified.csv')
    x_data = data[["SepalLengthCm","SepalWidthCm","PetalLengthCm","PetalWidthCm","Species", "Gender"]]
    y_data = data[["Species"]]
    return x_data, y_data

x_data, y_data = load_dataset()

# Preprocess -> normalize, encode
def normalize(x_data):
    Sc = MinMaxScaler()
    Sc = Sc.fit(x_data)
    return Sc.transform(x_data)
def ordinal_encoder(x_data):
    scaler = OrdinalEncoder()
    scaler = scaler.fit(x_data)
    return scaler.transform(x_data)
def encode(y_data):
    encoder = OneHotEncoder(sparse=False)
    encoder = encoder.fit(y_data)
    y_data = encoder.transform(y_data)
    return y_data

x_normalize = x_data[['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']]
x_string = x_data[['Gender']]
x_normalize = normalize(x_normalize)
x_string = ordinal_encoder(x_string)
x_data = np.column_stack((x_normalize, x_string))

y_data = encode(y_data)
print(x_data)


# Initialize weight, bias -> random
layer_count = {
    'input': 5,
    'hidden': 5,
    'output': 3
}
weight = {
    'hidden': tf.Variable(tf.random_normal([ layer_count['input'], layer_count['hidden'] ])),
    'output': tf.Variable(tf.random_normal([ layer_count['hidden'], layer_count['output'] ]))
}
bias = {
    'hidden': tf.Variable(tf.random_normal([ layer_count['hidden'] ])),
    'output': tf.Variable(tf.random_normal([layer_count['output'] ]))
}

# Optimizer
learning_rate = 0.1
number_epoch = 5000

def predict():
    # input -> hidden
    y1 = tf.matmul(x_input, weight['hidden']) + bias['hidden']

    # hidden -> output
    y2 = tf.matmul(y1, weight['output']) + bias['output']
    y2 = tf.nn.sigmoid(y2)

    return y2

x_train, x_test, y_train, y_test = train_test_split(x_data, y_data)

x_input = tf.placeholder(tf.float32, [None, layer_count['input']])
y_target = tf.placeholder(tf.float32, [None, layer_count['output']])

y_prediction = predict()
# Mean Square Error (MSE) -> 1/2 (y_target - y_prediction)**2
loss = tf.reduce_mean(0.5 * (y_target - y_prediction)**2)
optimizer = tf.train.GradientDescentOptimizer(learning_rate)
train = optimizer.minimize(loss)

# Train
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(number_epoch):
        # Train
        sess.run(train, feed_dict={x_input: x_train, y_target: y_train})
        if i % 500 == 0:
            match = tf.equal(tf.argmax(y_target, axis=1), tf.argmax(y_prediction, axis=1))
            matches = sess.run(match, feed_dict={
                x_input : x_test,
                y_target : y_test
            })
            print(matches)
            accuracy = tf.reduce_mean(tf.cast(match, tf.float32))
            print('Iteration {}, Accuracy {}'.format(i, 
            sess.run(accuracy, feed_dict={
                x_input : x_test,
                y_target : y_test
            })))
