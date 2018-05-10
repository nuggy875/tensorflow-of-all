# 12. data loading method
#
import tensorflow as tf
import numpy as np

xy = np.loadtxt('./data/data-01-test-score.csv', delimiter=',', dtype=np.float32)
# data setting #
# former ':' is about all instance, and letter ':'is about one instance
x_data = xy[:, 0:-1]
y_data = xy[:, [-1]]

# check the data
# print(x_data.shape, x_data, len(x_data))
# print(y_data.shape, y_data)

# build the graph #
# placeholder is always feed_dict
X = tf.placeholder(tf.float32, shape=[None, 3])
Y = tf.placeholder(tf.float32, shape=[None, 1])
# update various value
W = tf.Variable(tf.random_normal([3, 1]), name='weight')
b = tf.Variable(tf.random_normal([1]), name='bias')

# hypothesis
hypothesis = tf.matmul(X, W) + b

# cost/loss function
cost = tf.reduce_mean(tf.square(hypothesis - Y))

# optimizer
optimizer = tf.train.GradientDescentOptimizer(learning_rate=1e-5)
train = optimizer.minimize(cost)

# process graph #
sess = tf.Session()
sess.run(tf.global_variables_initializer())

for step in range(2001):
    cost_val, hy_val, _ = sess.run([cost, hypothesis, train], feed_dict={X: x_data, Y: y_data})

    if step % 10 == 0:
        print(step, "Cost: ", cost_val, "\mPrediction:\n", hy_val)


# Ask my score
print("Your score will be ", sess.run(hypothesis, feed_dict={X: [[100, 70, 101]]}))
print("Other score will be ", sess.run(hypothesis, feed_dict={X: [[60, 70, 110], [90, 100, 80]]}))
