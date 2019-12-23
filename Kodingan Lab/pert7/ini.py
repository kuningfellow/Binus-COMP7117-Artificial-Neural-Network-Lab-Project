import tensorflow as tf
class SOM:
    def __init__(self, width, height, input_dim):
        self.width = width
        self.height = height
        self.input_dim = input_dim

        self.num_node = width * height
        self.weight = tf.Variable(tf.random_normal([self.num_node, self.input_dim]))
        self.input = tf.placeholder(tf.float32, [self.input_dim])
        self.location = [tf.to_float([y, x]) for y in range(height) for x in range (width)]

        bmu = self.get_bmu()

        self.update_weight = self.update_neighbor(bmu)

    def update_neighbor(self, bmu):
        sigma = tf.to_float(tf.maximum(self.height, self.width)/2)
        
        distance = self.get_distance(self.location, bmu)

        ns = tf.exp(tf.div(tf.negative(tf.square(distance)), 2*tf.square(sigma)))

        lr = 0.1
        curr_lr = ns * lr

        stacked_lr = tf.stack([tf.tile(tf.slice(curr_lr, [i], [1]), [self.input_dim])for i in range(self.num_node)])

        xw_diff = tf.subtract(self.input, self.weight)

        delta_weight = tf.multiply(stacked_lr, xw_diff)

        new_weight = tf.add(self.weight, delta_weight)

        return tf.assign(self.weight, new_weight)


    def get_bmu(self):
        distance = self.get_distance(self.input, self.weight)
        bmu_index = tf.argmin(distance)
        bmu_location  = tf.to_float([tf.div(bmu_index, width), tf.mod(bmu_index, width)])
        return bmu_location
    
    def get_distance(self, node_a, node_b):
            squared_diff = tf.square(node_a - node_b)
            total_squared_diff = tf.reduce_sum(squared_diff, axis=1)
            distance=tf.sqrt(total_squared_diff)
            return distance

    def train(self, dataset, epoch):
        init = tf.global_variables_initializer()
        with tf.Session() as sess:
            sess.run(init)
            for i in range(epoch):
                for data in dataset:
                    sess.run(self.update_weight, feed_dict=
                    {self.input:data
                    })
            cluster = [[] for i in range(self.height)]
            location = sess.run(self.location)
            weight = sess.run(self.weight)
            for i, loc in enumerate(location):
                cluster[ int(loc[0])]=weight[i]
            
colors=[
        [0.,0.,0.],
        [1.,1.,1.],
        [1.,0.,0.],
        [0.,1.,0.],
        [0.,0.,1.],
        [1.,1.,0.],
        [1.,0.,1.],
        [0.,1.,1.],
    ]

width = 3
height = 3
input_dim  = 3
epoch = 1000

som = SOM(width, height, input_dim)
som.train(colors, epoch)

print('eta')