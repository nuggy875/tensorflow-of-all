# 16. classifying diabetes using several data
# 여러개의 data 를 이용해서 classifying 을 해 본다.

import tensorflow as tf
# set data #
# 1. create queue
# 2. create reader
# 3. record default

# input file name into the queue
filename_queue = tf.train.string_input_producer(['./data/data-03-diabetes.csv', './data/data-03-diabetes2.csv'], shuffle=False, name='filename_queue')

# set reader
reader = tf.TextLineReader()
key, value = reader.read(filename_queue)

# set default values
recode_defaults = [[0.], [0.], [0.], [0.], [0.], [0.], [0.], [0.], [0.]]
# decode value to csv type
xy = tf.decode_csv(value, record_defaults=recode_defaults)

# make batch
train_x_batch, train_y_batch = tf.train.batch([xy[0:-1], xy[-1:]], batch_size=10)

# build graph #
# placeholder for a tensor that will be always fed
X = tf.placeholder(tf.float32, shape=[None, 8])
Y = tf.placeholder(tf.float32, shape=[None, 1])
# Variable for a tensor that will be always update
W = tf.Variable(tf.random_normal([8, 1]))
b = tf.Variable(tf.random_normal([1]))
# hypothesis
hypothesis = tf.sigmoid(tf.matmul(X, W) + b)
# cost
cost = -tf.reduce_mean(Y * tf.log(hypothesis) + (1-Y) * tf.log(1 - hypothesis))
# optimizer
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
train = optimizer.minimize(cost)

# accuracy
predicted = tf.cast(hypothesis > 0.5, dtype=tf.float32)
accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, Y), dtype=tf.float32))

# launch the graph #
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    # 조정자의 역할을 한다.
    coord = tf.train.Coordinator()
    # 다중 연산을 위한 쓰레드 생성
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    for step in range(10001):
        x_batch, y_batch = sess.run([train_x_batch, train_y_batch])
        cost_val, hy_val, _ = sess.run([cost, hypothesis, train], feed_dict={X: x_batch, Y: y_batch})

        if step % 200 == 0:
            print(step, cost_val)

    # Accuracy report
    h, c, a = sess.run([hypothesis, predicted, accuracy], feed_dict={X: x_batch, Y: y_batch})
    print("\nHypothesis: ", h, "\nCorrect (Y): ", c, "\nAccuracy: ", a)

    coord.request_stop()
    coord.join(threads)
