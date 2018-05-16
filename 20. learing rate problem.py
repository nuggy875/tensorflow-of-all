# 20. leaning rate problem
import tensorflow as tf

# Set data #

# training data
x_data = [[1, 2, 1],
          [1, 3, 2],
          [1, 3, 4],
          [1, 5, 5],
          [1, 7, 5],
          [1, 2, 5],
          [1, 6, 6],
          [1, 7, 7]]
y_data = [[0, 0, 1],
          [0, 0, 1],
          [0, 0, 1],
          [0, 1, 0],
          [0, 1, 0],
          [0, 1, 0],
          [1, 0, 0],
          [1, 0, 0]]

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
X = tf.placeholder(tf.float32, shape=[None, 3])
Y = tf.placeholder(tf.float32, shape=[None, 3])

# variable for a tensor will be update
W = tf.Variable(tf.random_normal([3, 3]), 'weight')
b = tf.Variable(tf.random_normal([3]), 'bias')
# class number

# hypothesis
hypothesis = tf.nn.softmax(tf.matmul(X, W) + b)

# cost
cost = tf.reduce_mean(-tf.reduce_sum(tf.log(hypothesis) * Y, axis=1))

# minimize
# compare with small, big learning rate
small_learning_rate = 1e-10
big_learning_rate = 1.5
optimizer = tf.train.GradientDescentOptimizer(big_learning_rate).minimize(cost)


# Correct prediction test model 
prediction = tf.arg_max(hypothesis, 1)
is_correct = tf.equal(prediction, tf.arg_max(Y, 1))
accuray = tf.reduce_mean(tf.cast(is_correct, tf.float32))

# Launch graph #
with tf.Session() as sess:
    # init
    sess.run(tf.global_variables_initializer())
    for step in range(201):
        cost_val, W_val, _ = sess.run([cost, W, optimizer], feed_dict={X: x_data, Y:y_data})
        print(step, cost_val, W_val)

    # placeholder 를 이용해서 test, training 다 할 수 있다.
    # predict
    print("Prediction: ", sess.run(prediction, feed_dict={X: x_test}))
    # calculate the accuracy
    print("Accuracy: ", sess.run(accuray, feed_dict={X: x_test, Y: y_test}))

