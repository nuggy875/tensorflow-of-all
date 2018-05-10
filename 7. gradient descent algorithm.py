# 7. gradient descent algorithm
# 경사 하강법 이해를 목적으로 한다.

import tensorflow as tf
# 그래프 빌드 #
# 데이터
x_data = [1, 2, 3]
y_data = [1, 2, 3]
# 변하는 값
W = tf.Variable(tf.random_normal([1]), name='weight')
# 데이터를 담는 값 'feed_dict'
X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)
# Our hypothesis for linear model X * W --> simplify the numerical expression
hypothesis = X * W

# cost/loss function
cost = tf.reduce_mean(tf.square(hypothesis - Y))
# Launch the graph in a session

# Minimize : Gradient Descent using derivative: W -= Learning_rate * derivative
# gradient = d ( cost ) / dw
# W = W -  rate * d/dw( cost )
learning_rate = 0.1
gradient = tf.reduce_mean((W * X - Y) * X)
descent = W - learning_rate * gradient
update = W.assign(descent)
# assign : 연산 후 새 값을 재 설정하는 operator

sess = tf.Session()
# Initializes global variables in the graph.
sess.run(tf.global_variables_initializer())

for step in range(21):
    sess.run(update, feed_dict={X: x_data, Y: y_data})
    print(step, sess.run(cost, feed_dict={X: x_data, Y: y_data}), sess.run(W))


