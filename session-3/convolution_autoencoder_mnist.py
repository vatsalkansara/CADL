"""
Imports MNIST dataset
convolution autoencoder for MNIST dataset
"""
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from libs.utils import montage
from libs import gif
from libs.datasets import MNIST

ds=MNIST()
mean_img = np.mean(ds.X, axis=0)

# Let's get the first 1000 images of the dataset and reshape them
imgs = ds.X[:1000].reshape((-1, 28, 28))
# So the number of features is the second dimension of our inputs matrix, 784
n_features = ds.X.shape[1]
# And we'll create a placeholder in the tensorflow graph that will be able to get any number of n_feature inputs.
X = tf.placeholder(tf.float32, [None, n_features])
X_tensor = tf.reshape(X, [-1, 28, 28, 1])
n_filters = [16, 16, 16]
filter_sizes = [4, 4, 4]
current_input = X_tensor
# notice instead of having 784 as our input features, we're going to have
# just 1, corresponding to the number of channels in the image.
# We're going to use convolution to find 16 filters, or 16 channels of information in each spatial location we perform convolution at.
n_input = 1
# We're going to keep every matrix we create so let's create a list to hold them all
Ws = []
shapes = []
for layer_i, n_ouput in enumerate(n_filters):
	with tf.variable_scope("encoder/layer/{}".format(layer_i)):
		# we'll keep track of the shapes of each layer, As we'll need these for the decoder
		shapes.append(current_input.get_shape().as_list())
		W=tf.get_variable(name='W', shape=[filter_sizes[layer_i], filter_sizes[layer_i], n_input, n_ouput], initializer=tf.random_normal_initializer(mean=0.0, stddev=0.2))
		h=tf.nn.conv2d(current_input, W, strides=[1,2,2,1], padding='SAME')
		current_input = tf.nn.relu(h)
		Ws.append(W)
		n_input = n_ouput

# We'll first reverse the order of our weight matrices
Ws.reverse()
# and the shapes of each layer
shapes.reverse()
# and the number of filters (which is the same but could have been different)
n_filters.reverse()
# and append the last filter size which is our input image's number of channels
n_filters = n_filters[1:] + [1]
print(n_filters, filter_sizes, shapes)

# and then loop through our convolution filters and get back our input image
# we'll enumerate the shapes list to get us there
for layer_i, shape in enumerate(shapes):
	with tf.variable_scope("decoder/layer/{}".format(layer_i)):
		W = Ws[layer_i]
		h = tf.nn.conv2d_transpose(current_input, W,
            tf.stack([tf.shape(X)[0], shape[1], shape[2], shape[3]]),
            strides=[1, 2, 2, 1], padding='SAME')
		current_input = tf.nn.relu(h)

Y = current_input
Y = tf.reshape(Y, [-1, n_features])

cost = tf.reduce_mean(tf.reduce_mean(tf.squared_difference(X, Y), 1))
learning_rate = 0.001

# pass learning rate and cost to optimize
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)

# Session to manage vars/train
sess = tf.Session()
sess.run(tf.global_variables_initializer())

# Some parameters for training
batch_size = 100
n_epochs = 5

# We'll try to reconstruct the same first 100 images and show how
# The network does over the course of training.
examples = ds.X[:100]
# We'll store the reconstructions in a list
gif_imgs = []
for epoch_i in range(n_epochs):
    for batch_X, _ in ds.train.next_batch():
        sess.run(optimizer, feed_dict={X: batch_X - mean_img})
    recon = sess.run(Y, feed_dict={X: examples - mean_img})
    recon = np.clip((recon + mean_img).reshape((-1, 28, 28)), 0, 255)
    img_i = montage(recon).astype(np.uint8)
    gif_imgs.append(img_i)
    print(epoch_i)
gif.build_gif(imgs, saveto='conv-ae.gif', cmap='gray')