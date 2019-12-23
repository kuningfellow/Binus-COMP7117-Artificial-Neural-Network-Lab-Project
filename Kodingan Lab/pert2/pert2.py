import pandas as pd
import numpy as np
import random as rd
from sklearn.preprocessing import MinMaxScaler, OrdinalEncoder

raw_dataset = pd.read_csv('gender-dataset.csv')

print(raw_dataset)

x_data = raw_dataset[["Height", "Weight", "Shoe Size"]]
y_data = raw_dataset[["Gender"]]

scaler = MinMaxScaler()
scaler = scaler.fit(x_data)
x_data = scaler.transform(x_data)

Encoder = OrdinalEncoder()
Encoder = Encoder.fit(y_data)
y_data = Encoder.transform(y_data)

# print(y_data)

weight = np.random.normal(size=[3])
bias = np.random.normal(size=[1])

learning_rate = 0.1
number_epoch = 100

dataset = list(zip(x_data, y_data))
# print(dataset)

def step_function(y):
    if y >= 0:
        return 1
    else:
        return 0

def get_accuracy():
    correct_prediction = 0
    for x_sample, y_sample in dataset:
        # linear combination
        y = np.matmul(weight, x_sample) + bias
        # apply step function
        y = step_function(y)
        # hitung error
        err = y_sample - y
        if err == 0:
            correct_prediction += 1
    accuracy = correct_prediction*100 / len(dataset)
    return accuracy

# perceptron
# for i in range(number_epoch):
#     # milih random satu dari list
#     x_sample, y_sample = rd.choice(dataset)
#     # linear combination
#     y = np.matmul(weight, x_sample) + bias
#     # apply step function
#     y = step_function(y)
#     # hitung error
#     err = y_sample - y
#     #update weight, bias
#     weight = weight + (learning_rate * err * x_sample)
#     bias = bias + (learning_rate * err)

#     if (i+1) % 10 == 0:
#         # cek akurasi
#         print("Iterasi {}, Accuracy {}".format(i, get_accuracy()))

threshold = 1

def limit(y):
    if y>= threshold:
        return 1
    else:
        return 0

def get_accuracy_lms():
    correct_prediction = 0
    for x_sample, y_sample in dataset:
        # linear combination
        y = np.matmul(weight, x_sample) + bias
        # apply step function
        y = limit(y)
        # hitung error
        err = y_sample - y
        if err == 0:
            correct_prediction += 1
    accuracy = correct_prediction*100 / len(dataset)
    return accuracy

for i in range(number_epoch):
    # milih random satu dari list
    x_sample, y_sample = rd.choice(dataset)
    # linear combination
    y = np.matmul(weight, x_sample) + bias
    # apply step function -> linear
    y = y
    # hitung error -> 1/2 (y-t)**2
    # err = (y_sample - y)**2 /2
    #update weight, bias
    weight = weight - (learning_rate * (y - y_sample) * x_sample)
    bias = bias - (learning_rate * (y - y_sample))

    if (i+1) % 10 == 0:
        # cek akurasi
        print("Iterasi {}, Accuracy {}".format(i, get_accuracy()))
