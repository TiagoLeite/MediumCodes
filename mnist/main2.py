from matplotlib import pyplot as plt
import numpy as np

from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('MNIST_data', one_hot=True, validation_size=0)


def gen_image(arr):
    two_d = (np.reshape(arr, (28, 28)) * 255).astype(np.uint8)
    plt.imshow(two_d, interpolation='nearest', cmap='gray')
    return plt


# Get a batch of two random images and show in a pop-up window.
batch_xs = mnist.train.images[:20]

for k in range(20):
    gen_image(batch_xs[k]).savefig(str(k))
