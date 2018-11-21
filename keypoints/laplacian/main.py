import numpy as np
from PIL import Image
from scipy import signal
import  matplotlib.pyplot as plt
import sys


path = '/Users/franckthang/Work/PersonalWork/sift/resources/cat.jpg'
img = np.array(Image.open(path).convert('L'))
print("Image initial shape: {}".format(img.shape))


def blur(std, image):
    # print("Blur with {} std".format(std))
    print("Blur shape of image: {}".format(image.shape))
    def gaussian_matrix(x, y, std):
        std = 2 * (std ** 2)
        exp_arg = (x ** 2 + y ** 2) / std
        return (1 / (np.pi * std)) * np.exp([-exp_arg])[0]

    # Creating here the gaussian matrix
    kernel_shape = (9, 9)
    mid = (kernel_shape[0] - 1) / 2
    my_gaussian_matrix = np.zeros(kernel_shape)
    for h in range(kernel_shape[0]):
        for w in range(kernel_shape[1]):
            my_gaussian_matrix[h,w] = gaussian_matrix(np.abs(mid - h),np.abs(mid - w), std)

    # Normalize so that the sum is equal to 1
    my_gaussian_matrix /= np.sum(my_gaussian_matrix)
    blurred = signal.convolve2d(image, my_gaussian_matrix, 'same').astype('uint8')
    return blurred


def scale_space(img):
    octaves = 4
    std = np.sqrt(5)
    k = np.sqrt(0.5)
    pictures = 5
    j = 1
    image = img

    for octave in range(octaves):
        image = Image.fromarray(image)
        image = image.resize((image.size[0]//2,image.size[1]//2))
        image = np.array(image) 
        for i in range(pictures):
            plt.subplot(octaves, pictures, j)
            new_std = std * np.power(k, i)
            plt.imshow(blur(new_std, image), cmap="gray")
            j += 1

    plt.show()


scale_space(img)
