import tensorflow as tf
from matplotlib import pyplot as plt
import classification as cf
import clustering as cl

data = cf.DataSet()
data.getData('dataset/classification.csv')
data.performPCA()
data.splitData()

ann = cf.ANN([5, 14, 7], ['', 'sigmoid', 'sigmoid'])
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

in_data = cl.get_processed_data("dataset/clustering.csv")
width = 9
height = 9
input_dim = 3
epoch = 5000

som = cl.SOM(width, height, input_dim)
som.train(in_data, epoch)
plt.imshow(som.cluster)
plt.colorbar()
plt.show()