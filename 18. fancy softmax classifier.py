# 18. fancy softmax classifier
# make fancy soft max classifier - animal classification
import tensorflow as tf
import numpy as np

# set data #
# predicting animal type based on various features
xy = np.loadtxt('./data/data-04-zoo.csv', delimiter=',', dtype=np.float32)
x_data = xy[:, 0:-1]
y_data = xy[:, [-1]]

nb_classes = 7 # 0 ~ 6

# Build graph #

# placeholder for a tensor will be always fed
X = tf.placeholder(tf.float32, shape=[None, 16]) # the number of element is 16
Y = tf.placeholder(tf.int32, shape=[None, 1]) # 0 ~ 6, shape=(?, 1)
# one hot encoding
# tf.one_hot --> if rank is n, improve the dimension
# [[0],[3],......] 이라고 생각해보자 이를 one hot 해버리면
# [[[1, 0, 0, 0, 0, 0, 0]],
#  [[0, 0, 0, 1, 0, 0, 0]],
#   ....
Y_one_hot = tf.one_hot(Y, nb_classes) # one hot shape = (?, 1, 7)

# reshape --> -1 : everything
# reduce the dimension
# [[1, 0, 0, 0, 0, 0, 0],
#  [0, 0, 0, 1, 0, 0, 0],
#  ...
Y_one_hot = tf.reshape(Y_one_hot, [-1, nb_classes])

# variable for a tensor will be update
W = tf.Variable(tf.random_normal([16, nb_classes]), name='weight')
b = tf.Variable(tf.random_normal([nb_classes]), name='bias')

logits = tf.matmul(X, W) + b
# hypothesis
hypothesis = tf.nn.softmax(logits)
# ccost / loss
cost_i = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=Y_one_hot)
cost = tf.reduce_mean(cost_i)
# minimize
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(cost)

# predict
# argmax --> scoring
prediction = tf.argmax(hypothesis, 1)
correct_prediction = tf.equal(prediction, tf.argmax(Y_one_hot, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# Launch the graph
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for step in range(2001):
        sess.run(optimizer, feed_dict={X: x_data, Y: y_data})
        if step % 100 == 0:
            loss, acc = sess.run([cost, accuracy], feed_dict={X: x_data, Y: y_data})
            print("Step: {:5}\tLoss: {:.3f}\tAcc: {:.2%}".format(step, loss, acc))

    # Let's see if we can predict
    pred = sess.run(prediction, feed_dict={X: x_data})
    # y_data: (N, 1) = flatten => (N, ) matches pred.shape
    for p, y in zip(pred, y_data.flatten()):
        print("[{}] Prediction: {} True Y: {}".format(p == int(y), p, int(y)))
