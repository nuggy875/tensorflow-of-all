# 8. minimize cost function
# gradient descent 방법으로 cost를 최소화 하는 것을 목적으로 한다. W를 입력 받도록 코딩

import tensorflow as tf
# 입력
w_data = input()
# 그래프 빌드 #
x_data = [1, 2, 3]
y_data = [1, 2, 3]

w_data = float(w_data)
W = tf.Variable(w_data)
X = tf.placeholder(tf.float32, shape=[None])
Y = tf.placeholder(tf.float32, shape=[None])

# hypothesis function
hypothesis = X * W
# cost function
cost = tf.reduce_mean(tf.square(hypothesis - Y))
# minimize
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1)
train = optimizer.minimize(cost)

# 그래프 실행 #
sess = tf.Session()
# 초기화
sess.run(tf.global_variables_initializer())

for step in range(10):
    print(step, sess.run(W, feed_dict={X: x_data, Y: y_data}))
    sess.run(train ,feed_dict={X: x_data, Y: y_data})
