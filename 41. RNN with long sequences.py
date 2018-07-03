# 41. RNN with long sequences
import tensorflow as tf
import numpy as np

# Set Data #
# better data creation
sample = " if you want you"

# index -> char
# 각각의 idx 에 char 를 랜덤으로 부여하는 과정
idx2char = list(set(sample))
num_classes = len(idx2char)

# char -> index
# 랜덤으로 뽑힌 것들을 열거해서 하나씩 fix 하는 부분
char2idx = {c: i for i, c in enumerate(idx2char)}
# dict 의 key 가 문자가 되는부분
sample_idx = [char2idx[c] for c in sample]

# slice 를 통해서 data 를 만드는 부분
x_data = [sample_idx[:-1]]
y_data = [sample_idx[1:]]

# hyper parameters
dic_size = len(char2idx)
rnn_hidden_size = len(char2idx)
num_classes = len(char2idx)
batch_size = 1
sequence_length = len(sample) - 1
learning_rate = 0.1

X = tf.placeholder(dtype=tf.int32, shape=[None, sequence_length])
Y = tf.placeholder(dtype=tf.int32, shape=[None, sequence_length])
X_one_hot = tf.one_hot(X, num_classes)


# build the graph #

# cell creation
cell = tf.contrib.rnn.BasicLSTMCell(num_units=rnn_hidden_size, state_is_tuple=True)
# 0으로 초기화
initial_state = cell.zero_state(batch_size=batch_size, dtype=tf.float32)
# Drive the cell
outputs, _states = tf.nn.dynamic_rnn(cell=cell, inputs=X_one_hot, initial_state=initial_state, dtype=tf.float32)
# weight 생성
weights = tf.ones([batch_size, sequence_length])
# loss
sequence_loss = tf.contrib.seq2seq.sequence_loss(logits=outputs, targets=Y, weights=weights)
loss = tf.reduce_mean(sequence_loss)
train = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

prediction = tf.argmax(outputs, axis=2)

# Launch graph #

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(3000):
        l, _ = sess.run([loss, train], feed_dict={X: x_data, Y: y_data})
        result = sess.run(prediction, feed_dict={X: x_data})
        print(i, "loss:", l, "prediction: ", result, "true Y: ", y_data)

        # print char using dic
        result_str = [idx2char[c] for c in np.squeeze(result)]
        print(i, "loss:", l, "/tPrediction: ", ''.join(result_str))
