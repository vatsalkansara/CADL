import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

plt.style.use('ggplot')

from libs import utils

files=utils.get_celeb_files()

img=plt.imread(files[50])
# print(img)
# plt.imshow(img)
# plt.show()
# print(img.shape)

# plt.imshow(img[:, :, 0], cmap='gray')
# plt.show()
# plt.imshow(img[:, :, 1], cmap='gray')
# plt.imshow(img[:, :, 2], cmap='gray')

imgs = utils.get_celeb_imgs()
# plt.imshow(imgs[0])
# plt.show()
print(imgs[0].shape)

data = np.array(imgs)
print(data.shape)

mean_img = np.mean(data, axis=0)
# plt.imshow(mean_img.astype(np.uint8))
std_img = np.std(data, axis=0)
# plt.imshow(std_img.astype(np.uint8))
# plt.show()

flattened = data.ravel()

# plt.hist(flattened.ravel(), 255)
plt.show()
# plt.hist(mean_img.ravel(), 255)
# plt.show()

bins = 20
fig, axs = plt.subplots(1, 3, figsize=(12, 6), sharey=True, sharex=True)
axs[0].hist((data[0] - mean_img).ravel(), bins)
axs[0].set_title('(img - mean) distribution')
axs[1].hist((std_img).ravel(), bins)
axs[1].set_title('std deviation distribution')
axs[2].hist(((data[0] - mean_img) / std_img).ravel(), bins)
axs[2].set_title('((img - mean) / std_dev) distribution')
plt.show()

x = np.linspace(-3,3,100)
print(x)
print(x.shape)
print(x.dtype)

x = tf.linspace(-3.0,3.0,100)
print(x)
g= tf.get_default_graph()
print([op.name for op in g.get_operations()])
g.get_tensor_by_name('LinSpace' + ':0')

sess = tf.Session()
# Now we tell our session to compute anything we've created in the tensorflow graph.
computed_x = sess.run(x)
print(computed_x)

# Alternatively, we could tell the previous Tensor to evaluate itself using this session:
computed_x = x.eval(session=sess)
print(computed_x)

# We can close the session after we're done like so:
sess.close()
