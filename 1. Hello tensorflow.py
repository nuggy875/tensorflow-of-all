# 1. hello tensorflow

import tensorflow as tf
# Create a constant op
# This op is added as a node to the default graph
hello = tf.constant("Hello, Tensorflow!")

# start a TF session
sess = tf.Session()

# run the op and got result
print(sess.run(hello))

# 결과가 b가 나올때가 있는데 이것은 bytes string 이라는 뜻
