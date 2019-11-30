# Load the os library
import os
# Load the request module
import urllib.request
# import SSL which we neeed to setup for talking to the https server
import ssl
ssl._create_default_https_context = ssl._create_unverified_context
# for loading an image
import matplotlib.pyplot as plt
# Working with numerical data
import numpy as np
# For resizing the image
from scipy.misc import imresize

def plot_image(filename):
	img = plt.imread(filename)
	plt.imshow(img)
	plt.show()

def imcrop_tosquare(img):
    """Make any image a square image.

    Parameters
    ----------
    img : np.ndarray
        Input image to crop, assumed at least 2d.

    Returns
    -------
    crop : np.ndarray
        Cropped image.
    """
    if img.shape[0] > img.shape[1]:
        extra = (img.shape[0] - img.shape[1])
        if extra % 2 == 0:
            crop = img[extra // 2:-extra // 2, :]
        else:
            crop = img[max(0, extra // 2 + 1):min(-1, -(extra // 2)), :]
    elif img.shape[1] > img.shape[0]:
        extra = (img.shape[1] - img.shape[0])
        if extra % 2 == 0:
            crop = img[:, extra // 2:-extra // 2]
        else:
            crop = img[:, max(0, extra // 2 + 1):min(-1, -(extra // 2))]
    else:
        crop = img
    return crop


def imcrop(img, amt):
    if amt <= 0 or amt >= 1:
        return img
    row_i = int(img.shape[0] * amt) // 2
    col_i = int(img.shape[1] * amt) // 2
    return img[row_i:-row_i, col_i:-col_i]


if os.path.isdir('celeb_images'):
	pass
else:
	# Create a directory
	os.mkdir('celeb_images')
	# Now perform the following ten times
	for img_i in range(1,11):

		# Create a string using the current loop counter
		f = '000%03d.jpg' % img_i
		# and get the url with that string appended to the end
		url = 'https://s3.amazonaws.com/cadl/celeb-align/' + f
		# We'll print this out to the console so we can see how far we've gone
		print(url, end='\r')
		# And now download the url to location inside our new directory
		urllib.request.urlretrieve(url, os.path.join('celeb_images',f))

	print('\n')

files = [os.path.join('celeb_images',file_i)
		 for file_i in os.listdir('celeb_images')
		 if file_i.endswith('.jpg')]

print(files[0])

img = plt.imread(files[0])
# plt.imshow(img)
# plt.show()

# plt.figure()
# plt.imshow(img[:,:,0])
# plt.figure()
# plt.imshow(img[:,:,1])
# plt.figure()
# plt.imshow(img[:,:,2])
# plt.show()

img.astype(np.float32)

#print(np.random.randint(0, len(files)))
#f = files[np.random.randint(0, len(files))]
#plot_image(f)

square = imcrop_tosquare(img)
crop = imcrop(square,0.2)
rsz = imresize(crop, (64, 64))
# plt.imshow(rsz, interpolation='nearest');
mean_img = np.mean(rsz, axis=2)
print(mean_img.shape)
plt.imshow(mean_img, cmap='gray')
plt.show()

imgs = []
for file_i in files:
    img = plt.imread(file_i)
    square = imcrop_tosquare(img)
    crop = imcrop(square, 0.2)
    rsz = imresize(crop, (64, 64))
    imgs.append(rsz)
print(len(imgs))

data = np.array(imgs)
print(data.shape)