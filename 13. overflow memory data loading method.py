# 13. overflow memory data loading method
# address overflow memory data loading method
import tensorflow as tf

# data setting #

# tf.train.string_input_producer : [filename list], shuffle, name
filename_queue = tf.train.string_input_producer(
    ['./data/data-01-test-score.csv', './data/data-01-test-score2.csv'], shuffle=True, name='filename_queue')

# set reader
reader = tf.TextLineReader()
key, value = reader.read(filename_queue)

# default values, in case of empty columns. Also specifies the type of the decode result.
# if the recode rows is empty, this values is default
recode_defaults = [[0.], [0.], [0.], [0.]]
# decode value to csv type
xy = tf.decode_csv(value, record_defaults=recode_defaults)

# collect batches of cvs in
# tf.train.batch has so many
train_x_batch, train_y_batch = \
    tf.train.batch([xy[0:-1], xy[-1:]], batch_size=10)

# graph build #
# placeholder for a tensor that will be always fed
X = tf.placeholder(tf.float32, shape=[None, 3])
Y = tf.placeholder(tf.float32, shape=[None, 1])
# variable for a tensor that will update
W = tf.Variable(tf.random_normal([3, 1]))
b = tf.Variable(tf.random_normal([1]))

# hypothesis
hypothesis = tf.matmul(X, W) + b
# cost/loss function
cost = tf.reduce_mean(tf.square(hypothesis - Y))

# minimize
optimizer = tf.train.GradientDescentOptimizer(learning_rate=1e-5)
train = optimizer.minimize(cost)

# graph process #
sess = tf.Session()
# init
sess.run(tf.global_variables_initializer())

# start population the filename queue.
coord = tf.train.Coordinator()
threads = tf.train.start_queue_runners(sess=sess, coord=coord)

for step in range(2001):
    x_batch, y_batch = sess.run([train_x_batch, train_y_batch])
    cost_val, hy_val, _ = sess.run(
        [cost, hypothesis, train],
        feed_dict={X: x_batch, Y: y_batch})
    if step % 10 == 0:
        print(step, "Cost: ", cost_val, "\nPrediction\n", hy_val)

coord.request_stop()
coord.join(threads)

