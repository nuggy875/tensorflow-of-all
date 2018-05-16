# 22. using mnist data set
# mnist image is 28 * 28 pixel images
# so we need 784's weight\

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import random
import matplotlib.pyplot as plt

# Set data #
# Check out https://www.tensorflow.org/get_started/mnist/beginners
# for more information about the mnist dataset
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

nb_classes = 10
#MNIST data image of shape 28 * 28 = 784

# Build graph #
# placeholder for a tensor that will be always fed.
X = tf.placeholder(tf.float32, shape=[None, 784])
Y = tf.placeholder(tf.float32, shape=[None, nb_classes])
# variable for a tensor that will be updated.
W = tf.Variable(tf.random_normal([784, 10]), name='weight')
b = tf.Variable(tf.random_normal([nb_classes]), name='bias')
# hypothesis
hypothesis = tf.nn.softmax(tf.matmul(X, W) + b)
# cost/loss
cost = tf.reduce_mean(-tf.reduce_sum(Y * tf.log(hypothesis), axis=1))
# minimize
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1)
# train
train = optimizer.minimize(cost)

# test model
is_correct = tf.equal(tf.arg_max(hypothesis, 1), tf.arg_max(Y, 1))
# calculate accuracy
accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))

# Launch graph #
# parameters

# epoch is one pass of all the training set
training_epochs = 15
# batch is the number of learning set at once.
batch_size = 100
# iteration is total /batch_size

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for epoch in range(training_epochs):
        avg_cost = 0
        total_batch = int(mnist.train.num_examples/batch_size)
        for i in range(total_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            c, _ = sess.run([cost, train], feed_dict={X: batch_xs, Y: batch_ys})
            avg_cost += c / total_batch

        print('Epoch:', '%04d' % (epoch + 1), 'cost = ', '{:.9f}'.format(avg_cost))

    # Test the model using test sets
    # tensor.eval()
    print("Accuracy: ", accuracy.eval(session=sess, feed_dict={X: mnist.test.images, Y: mnist.test.labels}))

    # Get one and predict
    r = random.randint(0, mnist.test.num_examples - 1)
    print("Label: ", sess.run(tf.argmax(mnist.test.labels[r:r + 1], 1)))
    print("Prediction: ", sess.run(
        tf.argmax(hypothesis, 1), feed_dict={X: mnist.test.images[r:r + 1]}))

    plt.imshow(
        mnist.test.images[r:r + 1].reshape(28, 28),
        cmap='Greys',
        interpolation='nearest')
    plt.show()
