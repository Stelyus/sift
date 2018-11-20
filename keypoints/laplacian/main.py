import numpy as np
from PIL import Image
from scipy import signal
import  matplotlib.pyplot as plt
import sys

show = False

path = '/Users/franckthang/Work/PersonalWork/sift/resources/cat.jpg'
img = np.array(Image.open(path).convert('L'))
print("Image shape: {}".format(img.shape))



def blur(std):

    print("Blur with {} value".format(std))
    def gaussian_matrix(x, y, std):
        std = 2 * (std ** 2)
        exp_arg = (x ** 2 + y ** 2) / std
        return (1 / (np.pi * std)) * np.exp([-exp_arg])[0]

# Creating here the gaussian matrix
    kernel_shape = (7, 7)
    mid = (kernel_shape[0] - 1) / 2
    my_gaussian_matrix = np.zeros(kernel_shape)
    for h in range(kernel_shape[0]):
        for w in range(kernel_shape[1]):
            my_gaussian_matrix[h,w] = gaussian_matrix(np.abs(mid - h),np.abs(mid - w), std)

# Normalize so that the sum is equal to 1
    my_gaussian_matrix /= np.sum(my_gaussian_matrix)
    blurred = signal.convolve2d(img, my_gaussian_matrix, 'same').astype('uint8')

    if show:
        fig = plt.figure()
        ax = fig.add_subplot(121)
        ax1 = fig.add_subplot(122)
        ax.imshow(img, cmap="gray")
        ax.set_title("Original image")

        ax1.imshow(blurred, cmap="gray")
        ax1.set_title("Blurred image")
        plt.show()

    return blurred


def scale_space():
    octave = 4
    pictures = 5
    k = np.sqrt(2)
    std = np.sqrt(5)

    plt.subplot(2,3,1)
    plt.imshow(img, cmap="gray")

    for i in range(pictures):
        plt.subplot(2,3,i+2)
        std *= k
        plt.imshow(blur(std), cmap="gray")
    plt.show()

scale_space()
