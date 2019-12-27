from matplotlib import pyplot as plt
import tensorflow as tf
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np
class SOM:
    def __init__(self, width, height, input_dim):
        # initialize variables
        self.width = width
        self.height = height
        self.input_dim = input_dim
        self.num_node = width * height
        self.weight = tf.Variable(tf.random_normal([self.num_node, self.input_dim]))
        self.input = tf.placeholder(tf.float32, [self.input_dim])
        self.location = [ tf.to_float([y, x]) for y in range(height) for x in range(width) ]

        # Find best matching unit
        bmu = self.get_bmu()

        # Update its neightbors' weights
        self.update_weight = self.update_neighbor(bmu)

    def get_bmu(self):
        distance = self.get_distance(self.input, self.weight)
        bmu_index = tf.argmin(distance)
        bmu_location = tf.to_float([tf.div(bmu_index, self.width), tf.mod(bmu_index, self.width)])
        return bmu_location

    def update_neighbor(self, bmu):
        sigma = tf.to_float(tf.maximum(self.height, self.width) / 2)
        distance = self.get_distance(self.location, bmu)

        # Neightbor Strength
        ns = tf.exp( tf.div(tf.negative(tf.square(distance)), 2 * tf.square(sigma)))

        # Learning Rate
        lr = 0.1
        curr_lr = ns * lr

        stacked_lr = tf.stack( [ tf.tile(tf.slice(curr_lr, [i], [1]), [self.input_dim]) for i in range(self.num_node) ] )

        # Calculate input and weight diff
        xw_diff = tf.subtract(self.input, self.weight)
        delta_weight = tf.multiply(stacked_lr, xw_diff)

        # Update weight
        new_weight = tf.add(self.weight, delta_weight)

        return tf.assign(self.weight, new_weight)

    def get_distance(self, node_a, node_b):
        squared_diff = tf.square(node_a - node_b)
        total_squared_diff = tf.reduce_sum(squared_diff, axis=1)
        return tf.sqrt(total_squared_diff)

    def train(self, dataset, epoch):
        init = tf.global_variables_initializer()
        with tf.Session() as sess:
            sess.run(init)
            for i in range(epoch):
                for data in dataset:
                    sess.run(self.update_weight, feed_dict={
                        self.input: data
                    })
            cluster = [[] for i in range(self.height)]
            location = sess.run(self.location)
            weight = sess.run(self.weight)
            for i, loc in enumerate(location):
                cluster[int(loc[0])] = weight[i]
            self.cluster = cluster
    
"""
process the given data using PCA to reduce the dimension
"""
def get_processed_data(path_file):
    data = pd.read_csv(path_file)
    data = data.drop(columns = ["customer_id"])
    lis = list()
    for gender in data["gender"]:
        if(gender=="Male"):
            lis.append(0)
        else:
            lis.append(1)
    data["gender"] = lis
    data = MinMaxScaler().fit_transform(data)
    return PCA(n_components=3).fit_transform(data)

if __name__ == "__main__":
    in_data = get_processed_data("dataset/clustering.csv")
    width = 3
    height = 3
    input_dim = 3
    epoch = 5000
    som = SOM(width, height, input_dim)
    som.train(in_data, epoch)
    
    plt.imshow(som.cluster)
    plt.show()
    pass