# 23. tensor manipulation
import tensorflow as tf
import numpy as np

# 1차원 array #
t = np.array([0., 1., 2., 3., 4., 5., 6.])
# 1. rank
rank = t.ndim
print("rank :", rank)
# 2. shape
shape = t.shape
print("shape :", shape)
# 3. 원하는 위치의 김밥 가져오기 -1은 마지막
print(t[0],  t[1], t[-1])
# 4. 김밥을 몇개씩 먹고싶다.
print(t[2:5], t[4:-1])
# 5. 끝가지 다 먹고싶다.
print(t[:2], t[3:])

# 2차원 array #
# 1. 만들기
t = np.array([[1., 2., 3.], [4., 5., 6.], [7., 8., 9.], [10., 11., 12.]])
print(t)
# 2. rank
rank = t.ndim
print(rank)
# 3. shape
shape = t.shape
print(shape)

# Shape, Rank, Axis #

# Rank  -
# Shape -
t = tf.constant([1, 2, 3, 4])
print(tf.shape(t).eval(session=tf.Session()))

t = tf.constant([[1, 2],
                 [3, 4]])
print(tf.shape(t).eval(session=tf.Session()))

t = tf.constant([[[[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]],
                  [[13, 14, 15, 16], [17, 18, 19, 20], [21, 22, 23, 24]]]])
print(tf.shape(t).eval(session=tf.Session()))

# Axis 란?
# 4개의 축이 있다고 보면 된다. 가장 안쪽의 것부터 가장 큰값 바깥으로 나갈수록 0 or -1
# [                                             0
#    [                                          1
#        [                                      2
#            [1, 2, 3, 4],                      3
#            [5, 6, 7, 8],
#            [9, 10, 11, 12],
#        ],
#        [
#            [13, 14, 15, 16],
#            [17, 18, 19, 20],
#            [21, 22, 23, 24],
#        ]
#    ]
# ]

# matmul vs multiply
# 1. matmul
matrix1 = tf.constant([[1., 2.], [3., 4.]])
matrix2 = tf.constant([[1.], [2.]])
print(tf.matmul(matrix1, matrix2))

# 2. multiply
matrix1 = tf.constant([[1., 2.], [3., 4.]])
matrix2 = tf.constant([[1.], [2.]])
print(tf.multiply(matrix1, matrix2))

# broadcast 라는 개념 때문에 값이 다르다.
# shape 이 다르더라도 연산을 할 수 있도록 해주는 것

# 1. shape 이 같을 때
matrix1 = tf.constant([[3., 3.]])
matrix2 = tf.constant([[2., 2.]])
print(tf.multiply(matrix1, matrix2))

# 2. shape 이 다를 때 - 1
matrix1 = tf.constant([[1., 2.]])
matrix2 = tf.constant([[3.]])
print(tf.multiply(matrix1, matrix2))

# 3. shape 이 다를 때 - 2
matrix1 = tf.constant([[1., 2.]])
matrix2 = tf.constant([3., 4.])
print(tf.multiply(matrix1, matrix2))

# 3. shape 이 다를 때 - 3 : extend
matrix1 = tf.constant([[1., 2.]])
matrix2 = tf.constant([[3.], [4.]])
print(tf.multiply(matrix1, matrix2))

# 같은 shape로 연산 하는것이 좋다 #

# reduce_mean #
# 1. float 임을 주의
tf.reduce_mean([1, 2], axis=0)

x = [[1., 2.],
     [3., 4.]]
y = [[1., 2., 3.],
     [3., 4., 5.]]

# 2. 축없이 : 모두다 평균
tf.reduce_mean(x)

# 3. 축에 따라 평균
tf.reduce_mean(x, axis=0)
tf.reduce_mean(x, axis=1)

print(tf.reduce_mean(y, axis=0).eval(session=tf.Session()))
print(tf.reduce_mean(y, axis=1).eval(session=tf.Session()))

print(tf.reduce_mean(y, axis=-1).eval(session=tf.Session()))

# reduce_mean #
# 1. 축없이 : 모두다 더함
tf.reduce_sum(x)

# 2. 축에 따라 더함
tf.reduce_sum(x, axis=0)
tf.reduce_sum(x, axis=1)

print(tf.reduce_sum(y, axis=0).eval(session=tf.Session()))
print(tf.reduce_sum(y, axis=1).eval(session=tf.Session()))

print(tf.reduce_sum(y, axis=-1).eval(session=tf.Session()))

# 3. 적용
print(tf.reduce_mean(tf.reduce_sum(x, axis=-1)).eval(session=tf.Session()))

# arg_max + axis 개념 #
# 숫자가 아닌 index 로 생각 --> 위치를 구하는 것
x = [[0, 1, 2],
     [2, 1, 0]]
tf.argmax(x, axis=0)
tf.argmax(x, axis=1)
tf.argmax(x, axis=-1)

# ** Reshape ** #

# reshape
t = np.array([[[0, 1, 2],
               [3, 4, 5]],
              [[6, 7, 8],
               [9, 10, 11]]])
t.shape
tf.reshape(t, shape=[-1, 3])
# -1 알아서해 : 맨 안쪽의 값은 건드리지 않음
tf.reshape(t, shape=[-1, 1, 3])

# squeeze : 짜주는 것
tf.squeeze([[0], [1], [2]])
tf.expand_dims([0, 1, 2], 1)
# np.array([[0],[1],[2]])로 바뀜

# 't' is a tensor of shape [2]
tf.shape(tf.expand_dims(t, 0))  # [1, 2]
tf.shape(tf.expand_dims(t, 1))  # [2, 1]
tf.shape(tf.expand_dims(t, -1))  # [2, 1]

# 't2' is a tensor of shape [2, 3, 5]
tf.shape(tf.expand_dims(t2, 0))  # [1, 2, 3, 5]
tf.shape(tf.expand_dims(t2, 2))  # [2, 3, 1, 5]
tf.shape(tf.expand_dims(t2, 3))  # [2, 3, 5, 1]

# One hot #
# 숫자를 0, 1, 2, 3, 4, 5 이렇게 표현하지 않고
# 10000 010000, 001000, 000100, 000010, 000001 이렇게 바꿔주는 것 하나만 핫하게
# 자동적으로 automatically rank expand
# 전체 label의 갯수가 depth
tf.one_hot([[0], [1], [2], [0]], depth=3)
# [[[1, 0, 0]], [[0, 1, 0]], [[0, 0, 1]], [[1, 0, 0]]] rank 가 expand 됨
# reshape 하면 됨
tf.reshape(t, shape=[-1, 3])

# Casting #
# 캐스팅 하는 것
x = tf.constant([1.8, 2.2, 3.3, 4.9], dtype=tf.float32)
tf.cast(x, tf.int32)  # [1, 2], dtype=tf.int32

tf.cast([True, False, 1 == 1, 0 == 1], tf.int32)
# array([1, 0, 1, 0], dtype=int32)

# Stack #
x = tf.constant([1, 4])
y = tf.constant([2, 5])
z = tf.constant([3, 6])
tf.stack([x, y, z])  # [[1, 4], [2, 5], [3, 6]] (Pack along first dim.)
tf.stack([x, y, z], axis=1)  # [[1, 2, 3], [4, 5, 6]]

# ones_like #
tensor = tf.constant([[1, 2, 3], [4, 5, 6]])
tf.ones_like(tensor)  # [[1, 1, 1], [1, 1, 1]]

# zeros_like #
tensor = tf.constant([[1, 2, 3], [4, 5, 6]])
tf.zeros_like(tensor)  # [[0, 0, 0], [0, 0, 0]]

# zip #
# 복수개의 tensor 를 한번에 처리 하고 싶을 때
# 각각을 받아서 한번에 처리 할 수 있다.

for x, y in zip([1, 2, 3], [4, 5, 6]):
    print(x, y)
for x, y in zip([1, 2, 3], [4, 5, 6], [7, 8, 9]):
    print(x, y, z)

