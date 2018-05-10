# 11. multi-variable linear regression using matrix
# strength of using matrix is var의 갯수와 상관없이 쉽게 사용 할 수 있다. 

import tensorflow as tf
# data setting #
# x_data has 5 instance and 3 other var
# number of each element
x_data = [[73., 80., 75.],
          [93., 88., 93.],
          [89., 91., 90.],
          [96., 98., 100.],
          [73., 66., 70.]]

# x_data has 5 instance and just one value
y_data = [[152.],
          [185.],
          [180.],
          [196.],
          [142.]]

# build the graph #
X = tf.placeholder(tf.float32, shape=[None, 3])
Y = tf.placeholder(tf.float32, shape=[None, 1])

W = tf.Variable(tf.random_normal([3, 1]), name='weight')
b = tf.Variable(tf.random_normal([1]), name='bias')

# Hypothesis
hypothesis = tf.matmul(X, W) + b

# cost/loss function is same
cost = tf.reduce_mean(tf.square(hypothesis - Y))

# minimize : gradient optimizer
optimizer = tf.train.GradientDescentOptimizer(learning_rate=1e-5)
train = optimizer.minimize(cost)

# process the graph #
sess = tf.Session()
sess.run(tf.global_variables_initializer())

# learning...
for step in range(2001):
    cost_val, hy_val, _ = sess.run([cost, hy_val, train], feed_dict={X: x_data, Y: y_data})

    if step % 10 == 0:
        print(step, "Cost:", cost_val, "\nPrediction:\n", hy_val)