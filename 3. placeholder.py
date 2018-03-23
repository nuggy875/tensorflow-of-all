# 3. placeholder
# 실행시키는 단계에서 값들을 던져주고 싶을 때 만드는 노드중 하나
import tensorflow as tf

# 1 그래프 빌드
a = tf.placeholder(tf.float32)
b = tf.placeholder(tf.float32)
adder_node = a + b # + provides a shortcut for tf.add(a, b)

# 2 그래프 실행
sess = tf.Session()

# 3 리턴
# place holder 는 feed_dict 가 필수
print(sess.run(adder_node, feed_dict={a: 3, b: 4.5}))
print(sess.run(adder_node, feed_dict={a: [1, 3], b: [2, 4]}))

