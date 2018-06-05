# 34. mnist test using three layers and adam

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import random
import matplotlib.pyplot as plt

# Set data #
# Check out https://www.tensorflow.org/get_started/mnist/beginners
# for more information about the mnist dataset
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

# build graph #
X = tf.placeholder(dtype=tf.float32, shape=[None, 784])
Y = tf.placeholder(dtype=tf.float32, shape=[None, 10])

# model - 3 layers model #
# first layer
W1 = tf.get_variable("W1", shape=[784, 256], initializer=tf.contrib.layers.xavier_initializer())
b1 = tf.Variable(initial_value=tf.random_normal([256]))
L1 = tf.nn.relu(tf.matmul(X, W1) + b1)

# second layer
W2 = tf.get_variable("W2", shape=[256, 256], initializer=tf.contrib.layers.xavier_initializer())
b2 = tf.Variable(initial_value=tf.random_normal([256]))
L2 = tf.nn.relu(tf.matmul(L1, W2) + b2)

# third layer
W3 = tf.get_variable("W3", shape=[256, 10], initializer=tf.contrib.layers.xavier_initializer())
b3 = tf.Variable(initial_value=tf.random_normal([10]))
hypothesis = tf.matmul(L2, W3) + b3

# cost/ loss
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=hypothesis, labels=Y))
# minimize cost
optimizer = tf.train.AdamOptimizer(learning_rate=0.01)
train = optimizer.minimize(cost)

# 그런데 xavier 쓰고 싶은데?

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
    print('Learning Finished!')
    # tensor.eval()
    print("Accuracy: ", accuracy.eval(session=sess, feed_dict={X: mnist.test.images, Y: mnist.test.labels}))

    '''
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
    '''

