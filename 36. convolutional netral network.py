import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

# Set data #
# Check out https://www.tensorflow.org/get_started/mnist/beginners
# for more information about the mnist dataset
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
learning_rate = 0.001


# build graph #

# placeholder for a tensor will be always fed
X = tf.placeholder(dtype=tf.float32, shape=[None, 784])
X_img = tf.reshape(tensor=X, shape=[-1, 28, 28, 1])
Y = tf.placeholder(dtype=tf.float32, shape=[None, 10])

# Layer 1
# L1 image shape = (?, 28, 28, 1)
W1 = tf.Variable(initial_value=tf.random_normal(shape=[3, 3, 1, 32], stddev=0.01))
L1 = tf.nn.conv2d(input=X_img, filter=W1, strides=[1, 1, 1, 1], padding='SAME')
# Conv1 -> (?, 28, 28, 32)
L1 = tf.nn.relu(L1)
L1 = tf.nn.max_pool(value=L1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
# Pool1 -> (?, 14, 14, 32)

# Layer 2
# L2 input shape = (?, 14, 14, 32)
W2 = tf.Variable(initial_value=tf.random_normal(shape=[3, 3, 32, 64], stddev=0.01))
L2 = tf.nn.conv2d(input=L1, filter=W2, strides=[1, 1, 1, 1], padding='SAME')
# Conv2 -> (?, 14, 14, 64)
L2 = tf.nn.relu(L2)
L2 = tf.nn.max_pool(value=L2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
# Pool1 -> (?, 7, 7, 64)
# reshape for FC
L2 = tf.reshape(tensor=L2, shape=[-1, 7 * 7 * 64])

# Layer 3 - FC
W3 = tf.get_variable("W2", shape=[7 * 7 * 64, 10], initializer=tf.contrib.layers.xavier_initializer())
b = tf.Variable(initial_value=tf.random_normal([10]))

# hypothesis
hypothesis = tf.matmul(L2, W3) + b

# cost
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=hypothesis, labels=Y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

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
    print('Learning started. It takes sometime.')
    for epoch in range(training_epochs):
        avg_cost = 0
        total_batch = int(mnist.train.num_examples/batch_size)
        for i in range(total_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            c, _ = sess.run([cost, optimizer], feed_dict={X: batch_xs, Y: batch_ys})
            avg_cost += c / total_batch

        print('Epoch:', '%04d' % (epoch + 1), 'cost = ', '{:.9f}'.format(avg_cost))

    # Test the model using test sets
    print('Learning Finished!')
    # tensor.eval()
    print("Accuracy: ", accuracy.eval(session=sess, feed_dict={
        X: mnist.test.images, Y: mnist.test.labels}))
