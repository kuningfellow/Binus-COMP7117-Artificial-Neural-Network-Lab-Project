import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from matplotlib import pyplot as plt

def load_dataset():
    data = pd.read_csv('monthly-milk-production.csv', index_col='Month')
    return data

scaler = MinMaxScaler()
def normalize(dataset):
    global scaler
    scaler = scaler.fit(dataset)
    return scaler.transform(dataset)
def denrmalize(dataset):
    return scaler.inverse_transform(dataset)
def get_batches(dataset, num_batch, num_step):
    input_batch = np.zeros(shape=(num_batch, num_step, num_input))
    output_batch = np.zeros(shape=(num_batch, num_step, num_output))
    for i in range(num_batch):
        start = np.random.randint(0, len(dataset) - num_step)
        input_batch[i] = dataset[start : start + num_step]
        output_batch[i] = dataset[start+1 : start+1 + num_step]
    return input_batch, output_batch

num_input = 1
num_output = 1
num_context = 20

num_epoch = 10000
num_step = 12
num_batch = 4
learning_rate = 0.05

x_input = tf.placeholder(tf.float32, [None, num_step, num_input])
y_target = tf.placeholder(tf.float32, [None, num_step, num_output])

dataset = load_dataset()
dataset = normalize(dataset)

training_data = dataset[: -num_step]
testing_data = dataset[len(dataset) - num_step:]

cell = tf.nn.rnn_cell.BasicRNNCell(num_context, activation=tf.nn.sigmoid)
cell = tf.contrib.rnn.OutputProjectionWrapper(cell, output_size=num_output, activation=tf.nn.sigmoid)

prediction, state = tf.nn.dynamic_rnn(cell, x_input, dtype=tf.float32)

loss tf.reduce_mean(0.5 * (y_target-prediction) ** 2)
optimizer = tf.train.AdamOptimizer(learning_rate)
train = optimizer.minimze(loss)