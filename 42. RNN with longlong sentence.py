# 41. RNN with long sequences
import tensorflow as tf
import numpy as np

# Set Data #
# better data creation
sentence = ("if you want to build a ship, don't drum up people together to "
            "collect wood and don't assign them tasks and work, but rather "
            "teach them to long for the endless immensity of the sea.")


# 각각의 idx 에 char 를 랜덤으로 부여하는 과정
char_set = list(set(sentence))

# 랜덤으로 뽑힌 것들을 열거해서 하나씩 fix 하는 부분
char_dic = {w: i for i, w in enumerate(char_set)}

data_dim = len(char_set)
hidden_size = len(char_set)
num_classes = len(char_set)
sequence_length = 10  # Any arbitrary number
learning_rate = 0.1

dataX = []
dataY = []
for i in range(0, len(sentence) - sequence_length):
    x_str = sentence[i:i + sequence_length]
    y_str = sentence[i + 1: i + sequence_length + 1]
    print(i, x_str, '->', y_str)

    x = [char_dic[c] for c in x_str]  # x str to index
    y = [char_dic[c] for c in y_str]  # y str to index

    dataX.append(x)
    dataY.append(y)

batch_size = len(dataX)

X = tf.placeholder(dtype=tf.int32, shape=[None, sequence_length])
Y = tf.placeholder(dtype=tf.int32, shape=[None, sequence_length])

X_one_hot = tf.one_hot(X, num_classes)

# build the graph #

# cell creation
cell = tf.contrib.rnn.BasicLSTMCell(num_units=hidden_size, state_is_tuple=True)
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

#prediction = tf.argmax(outputs, axis=2)

# Launch graph #

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(3000):
        l, _, results = sess.run([loss, train, outputs], feed_dict={X: dataX, Y: dataY})
        print(" ")
        print(i, "loss:", l)

        for j, result in enumerate(results):
            index = np.argmax(result, axis=1)
            if j is 0:  # print all for the first result to make a sentence
                print(''.join([char_set[t] for t in index]), end='')
            else:
                print(char_set[index[-1]], end='')
