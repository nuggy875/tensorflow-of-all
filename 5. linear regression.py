# 5. linear regression
# 선형 회귀에 대하여 알아본다.

# X and Y data
import tensorflow as tf

# 학습을 할 data
#x_train = [1, 2, 3]
#y_train = [1, 2, 3]

# 그래프 빌드 #

# placeholder 사용
X = tf.placeholder(tf.float32, shape=[None])
Y = tf.placeholder(tf.float32, shape=[None])


# 계속 업데이트 되는 값은 Variable : trainable variable
# random_normal : 랜덤한 값을 준다. : parameter : shape
W = tf.Variable(tf.random_normal([1]), name='weight')
b = tf.Variable(tf.random_normal([1]), name='bias')

# Our hypothesis function is XW + b
hypothesis = X * W + b

# cost/loss function
cost = tf.reduce_mean(tf.square(hypothesis - Y))

# cost minimize - 일단은 지금 넘어감
optimizer = tf.train.GradientDescentOptimizer(learning_rate= 0.01)
train = optimizer.minimize(cost)
# train 노드

# 그래프 빌드 끝 #

# 그래프 실행 #

# launch the graph in a session
sess = tf.Session()
# initializes global variables in the graph - variable 을 사용할때면 무조건 초기화 필수
sess.run(tf.global_variables_initializer())

# Fit the line with new training data
for step in range(2001):
    cost_val, W_val, b_val, _ = \     # 값들을 집어 넣을 때
        sess.run([cost, W, b, train], # 한번에 돌리고 싶을대
                 feed_dict={X: [1, 2, 3, 4, 5],
                            Y: [2.1, 3.1, 4.1, 5.1, 6.1]})
    if step % 20 == 0:
        print(step, cost_val, W_val, b_val)

# Testing our model - 테스트 하는 부분
print(sess.run(hypothesis, feed_dict={X: [5]}))
print(sess.run(hypothesis, feed_dict={X: [2.5]}))
print(sess.run(hypothesis, feed_dict={X: [1.5, 3.5]})) # 동시에 대답해봐
