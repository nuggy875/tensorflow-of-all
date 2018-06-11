# 37. deep cnn
# to understand we make deep cnn network
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# set data #
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
# Check out https://www.tensorflow.org/get_started/mnist/beginners for
# more information about the mnist dataset

# hyper parameters
learning_rate = 0.001
training_epochs = 15
batch_size = 100

# 중간에 계속 바뀌는 값 이여야 하니까
# dropout(keep_prob) rate 0.7~0.5 on training, but should be 1 for testing
keep_prob = tf.placeholder(dtype=tf.float32)

# build graph #
# placeholder for a tensor will be always fed
X = tf.placeholder(dtype=tf.float32, shape=[None, 784])
X_img = tf.reshape(tensor=X, shape=[-1, 28, 28, 1])
Y = tf.placeholder(dtype=tf.float32, shape=[None, 10])

# layer1
W1 = tf.Variable(initial_value=tf.random_normal(shape=[3, 3, 1, 32], stddev=0.01))
L1 = tf.nn.conv2d(input=X_img, filter=W1, strides=[1, 1, 1, 1], padding='SAME')
L1 = tf.nn.relu(L1)
L1 = tf.nn.max_pool(value=L1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
L1 = tf.nn.dropout(x=L1, keep_prob=keep_prob)
# Conv -> (?, 28, 28, 1)
# Pool -> (?, 14, 14, 32)

# layer2
W2 = tf.Variable(initial_value=tf.random_normal([3, 3, 32, 64], stddev=0.01))
L2 = tf.nn.conv2d(input=L1, filter=W2, strides=[1, 1, 1, 1], padding='SAME')
L2 = tf.nn.relu(L2)
L2 = tf.nn.max_pool(value=L2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
L2 = tf.nn.dropout(x=L2, keep_prob=keep_prob)
# Conv -> (?, 14, 14, 32)
# Pool -> (?, 7, 7, 64)

# layer3
W3 = tf.Variable(initial_value=tf.random_normal([3, 3, 64, 128], stddev=0.01))
L3 = tf.nn.conv2d(input=L2, filter=W3, strides=[1, 1, 1, 1], padding='SAME')
L3 = tf.nn.relu(L3)
L3 = tf.nn.max_pool(value=L3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
L3 = tf.nn.dropout(x=L3, keep_prob=keep_prob)
# Conv -> (?, 7, 7, 64)
# Pool -> (?, 4, 4, 128)
L3_flat = tf.reshape(tensor=L3, shape=[-1, 128 * 4 * 4])

# layer4
W4 = tf.get_variable(name="W4", shape=[128 * 4 * 4, 625], initializer=tf.contrib.layers.xavier_initializer())
b4 = tf.Variable(initial_value=tf.random_normal([625]))
L4 = tf.nn.relu(tf.matmul(L3_flat, W4) + b4)
L4 = tf.nn.dropout(x=L4, keep_prob=keep_prob)
# (?, 2048)

#  layer5
W5 = tf.get_variable(name="W5", shape=[625, 10], initializer=tf.contrib.layers.xavier_initializer())
b5 = tf.Variable(initial_value=tf.random_normal([10]))
# hypothesis
hypothesis = tf.matmul(L4, W5) + b5

# cost
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=hypothesis, labels=Y))

# minimize
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    print('Learning started. It takes sometime.')
    for epoch in range(training_epochs):
        avg_cost = 0
        total_batch = int(mnist.train.num_examples / batch_size)

        for i in range(total_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size=batch_size)
            feed_dict = {X: batch_xs, Y: batch_ys, keep_prob: 0.7}
            c, _ = sess.run(fetches=[cost, optimizer], feed_dict=feed_dict)
            avg_cost += c / total_batch

        print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.9f}'.format(avg_cost))
    print('Learning Finished!')

    correct_prediction = tf.equal(tf.argmax(hypothesis, 1), tf.argmax(Y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    print('Accuracy:', sess.run(accuracy, feed_dict={
        X: mnist.test.images, Y: mnist.test.labels, keep_prob: 1}))