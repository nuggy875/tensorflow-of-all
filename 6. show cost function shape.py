# 6. show cost function
# cost function 이 가져야 하는 특성 ( convex function ) 을 이해하고 그 개형을 그려본다.

import tensorflow as tf
import matplotlib.pyplot as plt

# 그래프 빌드 #
# 데이터
X = [1, 2, 3]
Y = [1, 2, 3]
# 데이터를 담는 값
W = tf.placeholder(tf.float32)
# Our hypothesis for linear model X * W --> simplify the numerical expression
hypothesis = X * W

# cost/loss function
cost = tf.reduce_mean(tf.square(hypothesis - Y))
# Launch the graph in a session
# 그래프 실행 #
sess = tf.Session()
# Initializes global variables in the graph.
sess.run(tf.global_variables_initializer())

# 저장할 list를 만듬
# Variables for plotting cost function
W_val = []

cost_val = []
for i in range(-30, 50):
    feed_W = i * 0.1
    curr_cost, curr_W = sess.run([cost, W], feed_dict={W: feed_W})
    W_val.append(curr_W)
    cost_val.append(curr_cost)


# Show the cost function
plt.plot(W_val, cost_val)
plt.show()



