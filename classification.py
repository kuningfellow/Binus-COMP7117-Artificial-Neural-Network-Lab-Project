import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
import numpy as np

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

        self.train = self.optimizer.minimize(self.loss)

        # accuracy metric used = (total correct) / (total input)
        self.accuracy = tf.reduce_mean( tf.cast(tf.equal( tf.argmax(self.output,axis=1) , tf.argmax(self.target,axis=1) ), tf.float32) )


class DataSet:
    """
    Class for handling data
    """
    def __init__(self):
        super().__init__()

    def getData(self, path):
        """
        Obtains dataset as object attribute and performs normalization
        """
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
            # "legs",   #I don't think this is ordinal
            "tail",
            "domestic",
            "catsize"
        ]]
        # in_data is boolean, so no need for normalizing

        # Encode legs column
        # Use OneHotEncoder because leg count has no ordinal relationship in determining an animal class
        self.in_data = np.column_stack(( self.in_data , OneHotEncoder(sparse=False).fit_transform( self.data[['legs']] ) ))

        # OneHotEncoder because animal classes has no ordinal relationship between one another
        self.out_data = OneHotEncoder(sparse=False).fit_transform( self.data[['class_type']] )

    def splitData(self):
        """
        Splits data according to project the requirement
        """
        # Use 70% of dataset as train data. Use remaining dataset as test data
        self.train_in_data, self.test_in_data, self.train_out_data, self.test_out_data = train_test_split(self.in_data, self.out_data, train_size=0.7)

        # Use 20% of train data as validation. Use the remaining as actual training data
        self.train_in_data, self.validation_in_data, self.train_out_data, self.validation_out_data = train_test_split(self.train_in_data, self.train_out_data, test_size=0.2)
    
    def performPCA(self):
        """
        Performs PCA on in_data
        """
        # Take 5 component as per project request
        self.pca = PCA(n_components=5).fit(self.in_data)
        self.in_data = self.pca.transform(self.in_data)


data = DataSet()
data.getData('dataset/classification.csv')
data.performPCA()
data.splitData()

ann = ANN([5, 14, 7], ['', 'sigmoid', 'sigmoid'])
ann.createTrain(0.5)
"""
Considerations for our model
We believe that the feature 'legs' has no ordinal meaning in determining an animal's class type
Therefore we went ahead and encoded that feature as a boolean feature vector using OneHotEncoder

Input:
    We need 5 input nodes because we used PCA taking 5 components as features
Hidden:
    The original dataset is a linearly separable data.
    According to the Perceptron Theorem, if the dataset is linearly separable, then the perceptron learning rule will converge to weight vector that gives correct response for all training patters.
    So constructed a perceptron. We then used all the data without PCA as the training, validation, and testing.
    The training, validation, and testing all had 101 data entries and 17 features (11 from the requested features + 6 from expanding the 'legs' feature)
    And after 50000 epochs and a high learning rate (10, just for the purpose of quickly finding the minimum), it converged to 99.00990128517151% accuracy.
    We concluded that the dataset is linearly separable with it having 1 outlier. 100/101 * 100% = 99.009901%

    HOWEVER, with the requirement that the dataset be PCA'd. The reduced dimension dataset became NONLINEAR based on the previous testing method.
    Therefore we needed a hidden layer. Subsequent testing shows that there were no added benefits to using more than 1 hidden layer.
    So we decided to just use one hidden layer.
Output:
    We used 7 nodes for output because we believe that the classes has no ordinal relation and therefore should be considered as independent features.

The learning rate was set to 0.5 as a balance between faster minimum finding, and a low chance of overshooting
"""

# 5000 epochs as per project request
epoch = 5000
prevLoss = 0
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(epoch):
        sess.run(ann.train, feed_dict={
            ann.input: data.train_in_data,
            ann.target: data.train_out_data
        })
        if i % 100 == 99:
            curLoss = sess.run(ann.loss, feed_dict={ ann.input: data.validation_in_data, ann.target: data.validation_out_data })
            print("Epoch number {}, error = {}" . format( (i+1), curLoss ))
            if i % 500 == 499:
                # For every 500th epoch
                if curLoss < prevLoss:
                    # save model if better than pevious
                    saver = tf.train.Saver()
                    saver.save(sess, 'model/classification.cpkt')
                # get new validation error
                prevLoss = curLoss
    print("Accuracy : {}" . format( sess.run(ann.accuracy,feed_dict={ann.input:data.test_in_data, ann.target:data.test_out_data })*100 ))