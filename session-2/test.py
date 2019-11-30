import tensorflow as tf
import numpy as np

n_observations = 10
n_neurons = 2
xs = np.linspace(-3, 3, n_observations)
ys = np.sin(xs) + np.random.uniform(-0.5, 0.5, n_observations)

X = tf.convert_to_tensor(xs, np.float32)
Y = tf.convert_to_tensor(ys, np.float32)

sess = tf.Session()

W = tf.Variable(tf.random_normal([1, n_neurons], stddev=0.1))

b = tf.Variable(tf.constant(0, dtype=tf.float32, shape=[n_neurons]))

# x_expanded = sess.run(tf.expand_dims(X,1))
# print("x_expanded: ", x_expanded)

h = tf.matmul(tf.expand_dims(X, 1), W) + b
print("h shape: ", h)

Y_pred = tf.reduce_sum(h, 1)
# print(Y_pred)

sess.run(tf.global_variables_initializer())
W_computed = sess.run(W)
b_computed = sess.run(b)
h_computed = sess.run(h)
Y_pred_computed = sess.run(Y_pred)
print("W: ", W_computed)
print("b: ", b_computed)
print("h: ", h_computed)
print("Y_pred: ", Y_pred_computed)

sess.close()
