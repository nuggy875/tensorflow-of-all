import tensorflow as tf
import numpy as np

# hyper parameters
learning_rate = 0.1

# set data #
# unique chars    : h i e l o
# voc index (dic) : 0 1 2 3 4

idx2char = ['h', 'i', 'e', 'l', 'o']
x_data = [[0, 1, 0, 2, 3, 3]]     # h i h e l l
y_data = [[1, 0, 2, 3, 3, 4]]     # i h e l l o

x_one_hot = [[[1, 0, 0, 0, 0],    # h
              [0, 1, 0, 0, 0],    # i
              [1, 0, 0, 0, 0],    # h
              [0, 0, 1, 0, 0],    # e
              [0, 0, 0, 1, 0],    # l
              [0, 0, 0, 1, 0]]]   # l

input_dimension = 5
sequence_length = 6
hidden_size = 5
batch_size = 1

X = tf.placeholder(dtype=tf.float32, shape=[None, sequence_length, input_dimension])

# Y = tf.placeholder(dtype=tf.float32, shape=[None, sequence_length, hidden_size])
# Y 는 label 만 ?
Y = tf.placeholder(dtype=tf.int32, shape=[None, sequence_length])

# build the graph #
cell = tf.contrib.rnn.BasicLSTMCell(num_units=hidden_size, state_is_tuple=True)
initial_state = cell.zero_state(batch_size=batch_size, dtype=tf.float32)
# 0으로 초기화
outputs, _states = tf.nn.dynamic_rnn(cell=cell, inputs=X, initial_state=initial_state, dtype=tf.float32)

weights = tf.ones([batch_size, sequence_length])
sequence_loss = tf.contrib.seq2seq.sequence_loss(logits=outputs, targets=Y, weights=weights)
loss = tf.reduce_mean(sequence_loss)
train = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

prediction = tf.argmax(outputs, axis=2)

# Launch graph #
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(2000):
        l, _ = sess.run([loss, train], feed_dict={X: x_one_hot, Y: y_data})
        result = sess.run(prediction, feed_dict={X: x_one_hot})
        print(i, "loss:", l, "prediction: ", result, "true Y: ", y_data)

        # print char using dic
        result_str = [idx2char[c] for c in np.squeeze(result)]
        print("/tPrediction str: ", ''.join(result_str))
