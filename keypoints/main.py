import numpy as np
from PIL import Image
from scipy import signal
import  matplotlib.pyplot as plt
import sys


def laplacian_keypoints(img, std=1, show=False):

    def gaussian_matrix(x, y, std):
        std = 2 * (std ** 2)
        exp_arg = (x ** 2 + y ** 2) / std
        return (1 / (np.pi * std)) * np.exp([-exp_arg])[0]



    kernel_shape = (5,5) 
    mid = 2
    my_gaussian_matrix = np.zeros(kernel_shape)

    for h in range(kernel_shape[0]):
        for w in range(kernel_shape[1]):
            my_gaussian_matrix[h,w] = gaussian_matrix(np.abs(mid - h),np.abs(mid - w), std)

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




path = '/Users/franckthang/Work/PersonalWork/sift/resources/cat.jpg'
img = np.array(Image.open(path).convert('L'))
print("Image shape: {}".format(img.shape))

laplacian_keypoints(img, std=10, show=True)


