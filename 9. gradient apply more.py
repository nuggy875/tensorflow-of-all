# 9. gradient apply more
# 확장성을 가진 gradient 함수에 대하여 알아본다.

import tensorflow as tf
X = [1, 2, 3]
Y = [1, 2, 3]

# Set wrong model weights
W = tf.Variable(5.)
# Linear model
hypothesis = X * W
# cost function
cost = tf.reduce_mean(tf.square(hypothesis - Y))
# manual gradient
gradient = tf.reduce_mean((W * X - Y) * X) * 2

# Minimize cost function
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)

# get gradients
gvs = optimizer.compute_gradients(cost, [W])
# apply gradients
apply_gradients = optimizer.apply_gradients(gvs)

# Launch the graph in a session
sess = tf.Session()
sess.run(tf.global_variables_initializer())

for step in range(100):
    print(step, sess.run([gradient, W, gvs]))
    sess.run(apply_gradients)



