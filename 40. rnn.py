# 40. rnn

# text hihello
import tensorflow as tf

# set data #

# parameter
input_dimension = 5
sequence_length = 6
hidden_size = 5
batch_size = 1


# 우리가 가지고 있는 문자열
idx2char = ['h', 'i', 'e', 'l', 'o']

# Teach hello: hihell -> ihello

# Input data
x_data = [[0, 1, 0, 2, 3, 3]]   # hihell

x_one_hot = [[[1, 0, 0, 0, 0],   # h 0
              [0, 1, 0, 0, 0],   # i 1
              [1, 0, 0, 0, 0],   # h 0
              [0, 0, 1, 0, 0],   # e 2
              [0, 0, 0, 1, 0],   # l 3
              [0, 0, 0, 1, 0]]]  # l 3

# True
y_data = [[1, 0, 2, 3, 3, 4]]    # ihello

X = tf.placeholder(dtype=tf.float32, shape=[None, sequence_length, input_dimension])
Y = tf.placeholder(dtype=tf.int32, shape=[None, sequence_length])