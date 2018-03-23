#2. add
# 노드 2개를 더하는 graph를 만들어보자
import tensorflow as tf

# graph build

node1 = tf.constant(3.0, tf.float32)
node2 = tf.constant(4.0) # also tf.float32 implicitly
node3 = tf.add(node1, node2)
# node3 = node1 + node2

print("node1: ", node1, " node2: ", node2)
print("node3: ", node3)

# graph execute

sess = tf.Session()

# return 

print("sess.run(node1, node2): ", sess.run([node1, node2]))
print("sess.run(node3): ", sess.run(node3))