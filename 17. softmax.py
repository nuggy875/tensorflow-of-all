# 17. softmax function
# classify a number of class

import tensorflow as tf

# Set data #
x_data = [[1, 2, 1, 1],
          [2, 1, 3, 2],
          [3, 1, 3, 4],
          [4, 1, 5, 5],
          [1, 7, 5, 5],
          [1, 2, 5, 6],
          [1, 6, 6, 6],
          [1, 7, 7, 7]]

y_data = [[0, 0, 1],
          [0, 0, 1],
          [0, 0, 1],
          [0, 1, 0],
          [0, 1, 0],
          [0, 1, 0],
          [1, 0, 0],
          [1, 0, 0]]

# Build graph #
# placeholder for a tensor will be always fed
X = tf.placeholder(tf.float32, shape=[None, 4])
Y = tf.placeholder(tf.float32, shape=[None, 3])
nb_classes = 3

# variable for a tensor will be update
W = tf.Variable(tf.random_normal([4, nb_classes]), name='weight')
b = tf.Variable(tf.random_normal([nb_classes]), name='bias')

# tf.nn.softmax compute sofrmax activations
# softmax = exp(Logits) / reduce_sum(exp(Logits), dim)
# hypothesis
hypothesis = tf.nn.softmax(tf.matmul(X, W) + b)
# cost/loss : cross entropy
cost = tf.reduce_mean(-tf.reduce_sum(Y * tf.log(hypothesis), axis=1))
# minimize
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
train = optimizer.minimize(cost)

# Launch graph #
with tf.Session() as sess:
    #init
    sess.run(tf.global_variables_initializer())

    for step in range(2001):
        sess.run(train, feed_dict={X: x_data, Y: y_data})
        if step % 200 == 0:
            print(step, sess.run(cost, feed_dict={X: x_data, Y: y_data}))

    a = sess.run(hypothesis, feed_dict={X: [[1, 11, 7, 9]]})
    # arg_max : 가장 높은것
    # 축 : 1번째
    print(a, sess.run(tf.argmax(a, 1)))
    all = sess.run(hypothesis, feed_dict={X: [[1, 11, 7, 9],
                                              [1, 3, 4, 3],
                                              [1, 1, 0, 1]]})
    # arg_max : 가장 높은것
    # 축 : 1번째
    print(all, sess.run(tf.argmax(all, 1)))