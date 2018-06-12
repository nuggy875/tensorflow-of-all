# 38. cnn using python class
# using python class, we can simplify cnn code
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
# Check out https://www.tensorflow.org/get_started/mnist/beginners for
# more information about the mnist dataset

# hyper parameters
learning_rate = 0.001
training_epoch = 15
batch_size = 100


class Model:
    # 생성자 #
    def __init__(self, sess, name):
        self.sess = sess
        self.name = name
        self._build_net()

    # build graph #
    def _build_net(self):
        with tf.variable_scope(self.name):
            # dropout (keep_prob) rate 0.7~0.5 on training, but should be 1 for testing
            # self.~ 가 정의하는 방법인듯
            self.keep_prob = tf.placeholder(dtype=tf.float32)

            # input place holder
            self.X = tf.placeholder(dtype=tf.float32, shape=[None, 784])
            # image 28X28X1
            X_img = tf.reshape(tensor=self.X, shape=[-1, 28, 28, 1])
            self.Y = tf.placeholder(dtype=tf.float32, shape=[None, 10])

            # L1 Image shape = (?, 28, 28, 1)
            W1 = tf.Variable(tf.random_normal(shape=[3, 3, 1, 32], stddev=0.01))
            L1 = tf.nn.conv2d(input=X_img, filter=W1, strides=[1, 1, 1, 1], padding='SAME')
            L1 = tf.nn.relu(L1)
            L1 = tf.nn.max_pool(value=L1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
            L1 = tf.nn.dropout(L1, keep_prob=self.keep_prob)

            # L2 Image shape = (?, 14, 14, 32)
            W2 = tf.Variable(initial_value=tf.random_normal(shape=[3, 3, 32, 64], stddev=0.01))
            L2 = tf.nn.conv2d(input=L1, filter=W2, strides=[1, 1, 1, 1], padding='SAME')
            L2 = tf.nn.relu(L2)
            L2 = tf.nn.max_pool(value=L2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
            L2 = tf.nn.dropout(L2, keep_prob=self.keep_prob)

            # L3 Image shape = (?, 7, 7, 64)
            W3 = tf.Variable(initial_value=tf.random_normal(shape=[3, 3, 64, 128], stddev=0.01))
            L3 = tf.nn.conv2d(input=L2, filter=W3, strides=[1, 1, 1, 1], padding='SAME')
            L3 = tf.nn.relu(L3)
            L3 = tf.nn.max_pool(value=L3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
            L3 = tf.nn.dropout(L3, keep_prob=self.keep_prob)
            # Image shape = (?, 4, 4, 128)
            L3_flat = tf.reshape(tensor=L3, shape=[-1, 128 * 4 * 4])

            # L4 FC
            W4 = tf.get_variable(name="W4", shape=[128 * 4 * 4, 625],
                                 initializer=tf.contrib.layers.xavier_initializer())
            b4 = tf.Variable(initial_value=tf.random_normal(shape=[625]))
            L4 = tf.matmul(L3_flat, W4) + b4
            L4 = tf.nn.relu(L4)
            L4 = tf.nn.dropout(x=L4, keep_prob=self.keep_prob)

            # L5 FC
            W5 = tf.get_variable(name="W5", shape=[625, 10],
                                 initializer=tf.contrib.layers.xavier_initializer())
            b5 = tf.Variable(initial_value=tf.random_normal(shape=[10]))
            self.logits = tf.matmul(L4, W5) + b5

        # cost
        self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=self.Y))
        # optimizer
        self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(self.cost)

        # correct_prediction
        correct_prediction = tf.equal(tf.argmax(self.logits, 1), tf.argmax(self.Y, 1))
        # accuracy
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    def predict(self, x_test, keep_prop=1.0):
        return self.sess.run(self.logits, feed_dict={self.X: x_test, self.keep_prob: keep_prop})

    def get_accuracy(self, x_test, y_test, keep_prop=1.0):
        return self.sess.run(self.accuracy, feed_dict={self.X: x_test, self.Y: y_test, self.keep_prob: keep_prop})

    def train(self, x_test, y_test, keep_prop=0.7):
        return self.sess.run([self.cost, self.optimizer], feed_dict={
            self.X: x_test, self.Y: y_test, self.keep_prob: keep_prop})


# init
sess = tf.Session()
# constructor
m1 = Model(sess, 'm1')
sess.run(tf.global_variables_initializer())
print('Learning Started!')

# train my model
for epoch in range(training_epoch):
    avg_cost = 0
    total_batch = int(mnist.train.num_examples/batch_size)

    for i in range(total_batch):
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)
        c, _ = m1.train(batch_xs, batch_ys)
        avg_cost += c/total_batch

    print('Epoch:', '%04d' % (epoch + 1), 'cost=', '{:.9f}'.format(avg_cost))

print('Learning Finished!')

print('Accuracy:', m1.get_accuracy(mnist.test.images, mnist.test.labels))



























