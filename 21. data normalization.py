# 21.data normalization

# Non-normalized inputs
import tensorflow as tf
import numpy as np
from sklearn import preprocessing
# Set data #

# training data
xy = np.array([[828.659973, 833.450012, 908100, 828.349976, 831.659973],
               [823.02002, 828.070007, 1828100, 821.655029, 828.070007],
               [819.929993, 824.400024, 1438100, 818.97998, 824.159973],
               [816, 820.958984, 1008100, 815.48999, 819.23999],
               [819.359985, 823, 1188100, 818.469971, 818.97998],
               [819, 823, 1198100, 816, 820.450012],
               [811.700012, 815.25, 1098100, 809.780029, 813.669983],
               [809.51001, 816.659973, 1398100, 804.539978, 809.559998]])

### 이 부분을 통해 normalization 할 수 있다. ###
xy = preprocessing.minmax_scale(xy)
print(xy)

x_data = xy[:, 0:-1]
y_data = xy[:, [-1]]

# Evaluation our model using this test data set
# test data
x_test = [[2, 1, 1],
          [3, 1, 2],
          [3, 3, 4]]
y_test = [[0, 0, 1],
          [0, 0, 1],
          [0, 0, 1]]

# Build graph #

# placeholder for a tensor will be always fed
X = tf.placeholder(tf.float32, shape=[None, 4])
Y = tf.placeholder(tf.float32, shape=[None, 1])

# variable for a tensor will be update
W = tf.Variable(tf.random_normal([4, 1]), 'weight')
b = tf.Variable(tf.random_normal([1]), 'bias')
# class number

# hypothesis - linear regression
hypothesis = tf.matmul(X, W) + b

# cost
cost = tf.reduce_mean(tf.square(hypothesis - Y))

# minimize
optimizer = tf.train.GradientDescentOptimizer(learning_rate=1e-4)
train = optimizer.minimize(cost)

# Launch graph #
with tf.Session() as sess:
    # init
    sess.run(tf.global_variables_initializer())
    for step in range(2001):
        cost_val, hy_val, _ = sess.run([cost, hypothesis, train], feed_dict={X: x_data, Y: y_data})
        print(step, "Cost: ", cost_val, "\nPrecition:\n", hy_val)



