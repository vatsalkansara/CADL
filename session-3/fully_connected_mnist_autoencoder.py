"""
Imports MNIST dataset
Fully connected autoencoder for MNIST dataset
"""
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from libs.utils import montage
from libs import gif
from libs.datasets import MNIST

ds=MNIST()
mean_img = np.mean(ds.X, axis=0)

# print(ds.X.shape) # (70000, 784)
# plt.imshow(ds.X[0].reshape(28,28))
# plt.show()

# Let's get the first 1000 images of the dataset and reshape them
imgs = ds.X[:1000].reshape((-1, 28, 28))
# plt.imshow(montage(imgs), cmap='gray')
# plt.show()

dimensions = [512, 256, 128, 64]
# So the number of features is the second dimension of our inputs matrix, 784
n_features = ds.X.shape[1]

# And we'll create a placeholder in the tensorflow graph that will be able to get any number of n_feature inputs.
X = tf.placeholder(tf.float32, [None, n_features])

# let's first copy our X placeholder to the name current_input
current_input = X
n_input = n_features
# We're going to keep every matrix we create so let's create a list to hold them all
Ws = []
# We'll create a for loop to create each layer:
for layer_i, n_output in enumerate(dimensions):
	# just like in the last session,
    # we'll use a variable scope to help encapsulate our variables
    # This will simply prefix all the variables made in this scope
    # with the name we give it.
    with tf.variable_scope("encode/layer/{}".format(layer_i)):
    	# Create a weight matrix which will increasingly reduce
        # down the amount of information in the input by performing
        # a matrix multiplication
        W = tf.get_variable(name='W', shape=[n_input, n_output], initializer=tf.random_normal_initializer(mean=0.0, stddev=0.2))
        b = tf.get_variable(name='b', shape=[n_output], initializer=tf.constant_initializer(0.0))
        # Now we'll multiply our input by our transposed W matrix and add the bias
        h = tf.nn.bias_add(name='h', value=tf.matmul(current_input, W), bias=b)
        # And then use a relu activation function on its output
        current_input = tf.nn.relu(h)
        Ws.append(W)
        # We'll also replace n_input with the current n_output, so that on the
        # next iteration, our new number inputs will be correct.
        n_input = n_output

print(current_input.get_shape()) # (?, 64)

# We'll first reverse the order of our weight matrices
Ws = Ws[::-1]

# then reverse the order of our dimensions appending the last layers number of inputs.
dimensions = dimensions[::-1][1:] + [ds.X.shape[1]]
print(dimensions) # [128, 256, 512, 784]

for layer_i, n_output in enumerate(dimensions):
	# we'll use a variable scope again to help encapsulate our variables
    # This will simply prefix all the variables made in this scope
    # with the name we give it.
    with tf.variable_scope("decoder/layer/{}".format(layer_i)):
    	# Now we'll grab the weight matrix we created before and transpose it
        # So a 3072 x 784 matrix would become 784 x 3072
        # or a 256 x 64 matrix, would become 64 x 256
        W=tf.transpose(Ws[layer_i])
        b=tf.get_variable(name='b', shape=[n_output], dtype=tf.float32, initializer=tf.constant_initializer(0.0))
        h=tf.nn.bias_add(name='h', value=tf.matmul(current_input, W), bias=b)
        current_input=tf.nn.relu(h)
        n_input=n_output

Y = current_input
# We'll first measure the average difference across every pixel
cost = tf.reduce_mean(tf.squared_difference(X, Y), 1)
print(cost.get_shape()) #(?,)
cost = tf.reduce_mean(cost)
learning_rate = 0.001
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)
# We create a session to use the graph
sess = tf.Session()
sess.run(tf.global_variables_initializer())
# Some parameters for training
batch_size = 100
n_epochs = 50

# We'll try to reconstruct the same first 100 images and show how
# The network does over the course of training.
examples = ds.X[:100]
# We'll store the reconstructions in a list
gif_imgs = []
fig, ax = plt.subplots(1, 1)
for epoch_i in range(n_epochs):
    for batch_X, _ in ds.train.next_batch():
        sess.run(optimizer, feed_dict={X: batch_X - mean_img})
    recon = sess.run(Y, feed_dict={X: examples - mean_img})
    recon = np.clip((recon + mean_img).reshape((-1, 28, 28)), 0, 255)
    img_i = montage(recon).astype(np.uint8)
    gif_imgs.append(img_i)
    print(epoch_i) #, sess.run(cost, feed_dict={X: batch_X - mean_img}))
gif.build_gif(gif_imgs, saveto='ae.gif', cmap='gray')
