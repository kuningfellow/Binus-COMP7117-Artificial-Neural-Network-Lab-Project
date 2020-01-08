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
"""
Considerations for our model
We believe that the feature 'legs' has no ordinal meaning in determining an animal's class type
Therefore we went ahead and encoded that feature as a boolean feature vector using OneHotEncoder

Input:
    We need 5 input nodes because we used PCA taking 5 components as features
Hidden:
    We are not sure if the problem is a linear problem or not.
    According to the Perceptron Theorem, if the dataset is linearly separable, then the perceptron learning rule will converge to weight vector that gives correct response for all training patters.
    Using 50000 epoch and 0.5 learning rate, we can consistently get all except 1 data correctly.
    But using 200000 epoch and 0.01 learning rate, we sometimes get more than 1 data wrong.
    If we always get all except 1 data correctly, we might consider that data as an outlier, so the problem could be linearly separable.
    But testing results shows that convergence using a low learning rate is a hit and miss.

    HOWEVER, with the requirement that the dataset be PCA'd. The reduced dimension dataset became NONLINEAR based on the previous testing method.
    Even with 50000 epoch and 0.5 learning rate, we failed to produce the same result as before.
    Therefore it is a nonlinear problem and we needed a hidden layer.
    Subsequent testing shows that there were no added benefits to using more than 1 hidden layer.
    So we decided to just use one hidden layer.
Output:
    We used 7 nodes for output because we believe that the classes has no ordinal relation and therefore should be considered as independent features.

The learning rate was set to 0.5 as a balance between faster minimum finding, and a low chance of overshooting
"""

"""
For unit testing
"""
if __name__ == "__main__":
    data = DataSet()
    data.getData('dataset/classification.csv')
    data.performPCA()
    data.splitData()

    ann = ANN([5, 14, 7], ['', 'sigmoid', 'sigmoid'])
    ann.createTrain(0.5)
    # 5000 epochs as per project request
    epoch = 5000
    validationLoss = -1
    prevValidationLoss = -1
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for i in range(epoch):
            sess.run(ann.train, feed_dict={
                ann.input: data.train_in_data,
                ann.target: data.train_out_data
            })

            if i % 100 == 99:
                curLoss = sess.run(ann.loss, feed_dict={ ann.input: data.train_in_data, ann.target: data.train_out_data })
                print("Epoch number {}, error = {}" . format( (i+1), curLoss ))

            if i % 500 == 499:
                # For every 500th epoch
                validationLoss = sess.run(ann.loss, feed_dict={ ann.input: data.validation_in_data, ann.target: data.validation_out_data })
                if i == 500 or validationLoss < prevValidationLoss:
                    # save model if the first 500 or if better than pevious
                    saver = tf.train.Saver()
                    saver.save(sess, 'model/classification.cpkt')
                # get new validation error
                prevValidationLoss = validationLoss

        print("Accuracy : {}" . format( sess.run(ann.accuracy,feed_dict={ann.input:data.test_in_data, ann.target:data.test_out_data })*100 ))