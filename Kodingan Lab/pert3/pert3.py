import tensorflow as tf
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.model_selection import train_test_split

epoch = 3000
learning_rate = 0.2
layer_count = {
    'input' : 5,
    'hidden' : 8,
    'output': 3
}
weight = {
    'hidden' : tf.Variable(tf.random_normal([layer_count['input'], layer_count['hidden']])),
    'output' : tf.Variable(tf.random_normal([layer_count['hidden'], layer_count['output']]))
}

bias = {
    'hidden' : tf.Variable(tf.random_normal([layer_count['hidden']])),
    'output' : tf.Variable(tf.random_normal([layer_count['output']]))
}

x_input = tf.placeholder(tf.float32, [None, layer_count['input']])
y_target = tf.placeholder(tf.float32, [None, layer_count['output']])

# print(tf.run(c))

def load_dataset():
    raw_dataframe = pd.read_csv('Iris.csv')
    # input_data = raw_dataframe[['Id','SepalLengthCm','SepalWidthCm','PetalLengthCm','PetalWidthCm']]
    input_data = raw_dataframe.drop(columns=['Species'])
    target_data = raw_dataframe[['Species']]
    return input_data, target_data

def normalize(data):
    Sc = MinMaxScaler()
    Sc.fit(data)
    return Sc.transform(data)


def encode(data):
    En = OneHotEncoder(sparse=False)
    En.fit(data)
    return En.transform(data)

def predict():
    wx = tf.matmul(x_input, weight['hidden'])
    wx_b = wx +bias['hidden']
    y = tf.nn.sigmoid(wx_b)

    wx2 = tf.matmul(y, weight['output'])
    wx_b2 = wx2 + bias['output']
    y2 = tf.nn.sigmoid(wx_b2)

    return y2

input_data, target_data = load_dataset()

# preprocessing (Normalize, Encode)

input_data = normalize(input_data)
target_data = encode(target_data)


# Divide to train & test_data

input_train_data, input_test_data, target_train_data, target_test_data = train_test_split(input_data, target_data, test_size = 0.2)

# Prepare loss, prediction functions

prediction = predict()
loss = tf.reduce_mean(0.6 * (y_target - prediction)**2)
train = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)
# Run session

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)

    for i in range(epoch + 1):
        sess.run(train,
                feed_dict= {x_input : input_train_data,
                            y_target : target_train_data})
        if i % 200 == 0:
            matches = tf.equal(tf.argmax(prediction, axis=1), tf.argmax(y_target, axis=1))
            accuracy = tf.reduce_mean(tf.cast(matches, tf.float32))
            print('Iteration', i, " Accuracy: ", end='')
            print(sess.run(accuracy, feed_dict={
                x_input : input_test_data,
                y_target : target_test_data
            }))