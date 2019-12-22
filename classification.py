import pandas as pd
import tensorflow as tf

class ANN:
    """
    Single instance of ANN

    layers = list of integers indicating the number of nodes on each layer

    activations = list of strings indicating the activation function to be used on each layer    
    """

    def __init__(self, layers, activations):
        super().__init__()

        self.layers = list()    # self.layers[k] = number of nodes for the k-th layer
        self.biases = list()    # self.biases[k] = bias tensor for the k-th layer
        self.weights = list()   # self.weights[k] = weight tensor for the (k-1)-th layer to the k-th layer

        # input for this ANN
        self.input = tf.placeholder(tf.float32, [None, layers[0]])

        # output for this ANN
        self.output = self.input

        # target for this ANN
        self.target = tf.placeholder(tf.float32, [None, layers[len(layers) - 1]])

        # generate model
        prevLayerCount = 0
        prevLayer = self.output
        for layer, activation in zip(layers, activations):
            self.layers.append(layer)
            if len(self.layers) == 1:
                # at input layer, nothing to do
                self.weights.append(-1)
                self.biases.append(-1)
            else:
                # generate weight tensor
                tfWeight = tf.Variable(tf.random_normal( [prevLayerCount, layer] ))
                # generate bias tensor
                tfBias = tf.Variable(tf.random_normal( [layer] ))

                self.weights.append(tfWeight)
                self.biases.append(tfBias)

                # add layer to ANN
                self.output = tf.matmul(prevLayer, tfWeight) + tfBias
                if activation == "sigmoid":
                    self.output = tf.nn.sigmoid(self.output)

                prevLayer = self.output
            prevLayerCount = layer

    def createTrain(self, learning_rate):
        # mean squared error
        self.loss = tf.reduce_mean(0.5 * (self.target - self.output)**2)

        # use gradient descent optimizer
        self.optimizer = tf.train.GradientDescentOptimizer(learning_rate)

        self.train = optimizer.minimize(loss)

        # accuracy metric used = (total correct) / (total input)
        self.accuracy = tf.reduce_mean( tf.cast(tf.equal( tf.argmax(self.output,axis=1) , tf.argmax(self.target,axis=1) ), tf.float32) )


class DataSet:
    """
    Class for handling data
    """
    def __init__(self):
        super().__init__()
    
    def getData(self, path):
        self.data = pd.read_csv(path)
        self.in_data = self.data[[
            "hair",
            "feathers",
            "eggs",
            "milk",
            "toothed",
            "backbone",
            "breathes",
            "fins",
            "legs",
            "tail",
            "domestic",
            "catsize"
        ]]
        self.out_data = self.data[[
            "class_type"
        ]]
