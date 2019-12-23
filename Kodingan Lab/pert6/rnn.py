import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from matplotlib import pyplot as plt

def load_dataset():
    dataset = pd.read_csv('monthly-milk-production.csv', index_col='Month')
    return dataset

scaler = MinMaxScaler()
def normalize(dataset):
    global scaler
    scaler = scaler.fit(dataset)
    return scaler.transform(dataset)

def denormalize(dataset):
    return scaler.inverse_transform(dataset)

def get_batches(dataset, num_batch, num_step):
    input_batch = np.zeros(shape=(num_batch, num_step, num_input))
    target_batch = np.zeros(shape=(num_batch, num_step, num_output))
    for i in range(num_batch):
        start = np.random.randint(0, len(dataset)-num_step)
        input_batch[i] = dataset[start : start + num_step]
        target_batch[i] = dataset[start+1 : start+1 + num_step]
    return input_batch, target_batch

# initialize model variables
num_input = 1
num_output = 1
num_context = 20

# initialize training variables
num_epoch = 10000
num_step = 12
num_batch = 4
learning_rate = 0.05

# initialize placeholder
x_input = tf.placeholder(tf.float32, [None, num_step, num_input])       # num_step kali melihat ke belakang
y_target = tf.placeholder(tf.float32, [None, num_step, num_output])

# load dataset & normalize
dataset = load_dataset()
dataset = normalize(dataset)

# print(dataset)
# plt.plot(dataset)
# plt.show()

# split to train & test data
    # ambil data pertama hingga 12 terakhir
training_data = dataset[: -num_step]
    # ambil 12 data terakhi
testing_data = dataset[len(dataset) - num_step:]

# RNN cell && dynamic rnn
cell = tf.nn.rnn_cell.BasicRNNCell(num_context, activation=tf.nn.sigmoid)
cell = tf.contrib.rnn.OutputProjectionWrapper(cell, output_size=num_output, activation=tf.nn.sigmoid)

prediction, state = tf.nn.dynamic_rnn(cell, x_input, dtype=tf.float32)

# Loss, optimizer, train
loss = tf.reduce_mean(0.5 * (y_target-prediction) ** 2)
optimizer = tf.train.AdamOptimizer(learning_rate)
train = optimizer.minimize(loss)

# train data
init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    for i in range(num_epoch):
        input_batch, target_batch = get_batches(training_data, num_batch, num_step)
        sess.run(train, feed_dict={x_input: input_batch, y_target: target_batch})
        if (i+1) % 200 == 0:
            print('Iteration: {} Loss {}'.format(
                i+1,
                sess.run(
                    loss,
                    feed_dict = {
                        x_input: input_batch,
                        y_target: target_batch
                    }
                )
            ))

#    saver = tf.train.Saver()
#    saver.save(sess, 'model/rnn.ckpt')
#    saver.restore(sess, 'model/rnn.ckpt')

# test data
    previous_data = list(training_data)
    for i in range(12):
        x_batch = np.array(previous_data[-num_step:]).reshape((1, num_step, num_input))
        y_prediction = sess.run(prediction, feed_dict={
            x_input: x_batch,
        })
        y_prediction = y_prediction[0, -1]
        previous_data.append(y_prediction)

raw_dataset = load_dataset()
actual_data = dataset[-12:]
x_axis = raw_dataset.index[-12:]

actual_data = denormalize(actual_data)
previous_data = denormalize(previous_data)

# x, y, x1, y1
plt.plot(x_axis, actual_data, x_axis, previous_data[-12:])
plt.show()
