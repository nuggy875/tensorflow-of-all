# 4. variable
# 변수의 의미 - 업데이트 되는 값
import tensorflow as tf

# 그래프 빌드 #
# 타입과 shape
X = tf.placeholder(tf.float32, [None, 3])

# [3, 2] 의 shape 를 가지는 data
x_data = [[1, 2, 3], [4, 5, 6]]

# Variable : 계속해서 업데이트 되는 값
# random_normal : 정규분포에 따라 랜덤값이 만들어짐
W = tf.Variable(tf.random_normal([3, 2]))
b = tf.Variable(tf.random_normal([2, 1]))

# 모델 설정
expr = tf.matmul(X, W) + b

# 그래프 실행 #
# 세션 생성
sess = tf.Session()
# 초기화
sess.run(tf.global_variables_initializer())

# 그래프 업데이트 #
print("=== x_data ===")
print(x_data)
print("=== W ===")
print(sess.run(W))
print("=== b ===")
print(sess.run(b))
print("=== expr ===")
print(sess.run(expr, feed_dict={X: x_data}))

sess.close()