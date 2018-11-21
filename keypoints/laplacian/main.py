import numpy as np
from PIL import Image
from scipy import signal
import  matplotlib.pyplot as plt
import sys


path = '/Users/franckthang/Work/PersonalWork/sift/resources/cat.jpg'
img = np.array(Image.open(path).convert('L'))
print("Image initial shape: {}".format(img.shape))



def diff_gaussian(octaves, show=False):
    diff_gaussian = {n: [] for n in octaves}
    for key in octaves:
        pictures = octaves[key]
        for idx, picture in enumerate(pictures[:-1]):
            diff_gaussian[key].append(picture - pictures[idx + 1])

    if show:
        j = 1
        for key in diff_gaussian:
            for picture in diff_gaussian[key]:
                plt.subplot(len(diff_gaussian), len(diff_gaussian[key]), j)
                plt.imshow(picture, cmap="gray")
                j += 1
        plt.show()
    return diff_gaussian


def blur(std, image):
    # print("Blur with {} std".format(std))
    # print("Blur shape of image: {}".format(image.shape))
    def gaussian_matrix(x, y, std):
        std = 2 * (std ** 2)
        exp_arg = (x ** 2 + y ** 2) / std
        return (1 / (np.pi * std)) * np.exp([-exp_arg])[0]

    # Creating here the gaussian matrix
    kernel_shape = (5,5)
    mid = (kernel_shape[0] - 1) / 2
    my_gaussian_matrix = np.zeros(kernel_shape)
    for h in range(kernel_shape[0]):
        for w in range(kernel_shape[1]):
            my_gaussian_matrix[h,w] = gaussian_matrix(np.abs(mid - h),np.abs(mid - w), std)

    # Normalize so that the sum is equal to 1
    my_gaussian_matrix /= np.sum(my_gaussian_matrix)
    blurred = signal.convolve2d(image, my_gaussian_matrix, 'same').astype('uint8')
    return blurred



def scale_space(img, show=False):
    # Here for each octave we have the same std
    nb_octave = 4
    octaves = {n: [] for n in range(1,nb_octave+1)}
    pictures = 5
    scale = [np.power(x,2) for x in range(1, nb_octave+1)][::-1]
    image = img
    std = np.sqrt(.5)

    for octave in range(1, nb_octave):
        image = Image.fromarray(image)
        image = image.resize((image.size[0]//2,image.size[1]//2))
        image = np.array(image) 
        for i in range(pictures):
            new_std = std * np.power(np.sqrt(2), i)
            octaves[octave].append(blur(new_std, image))

    if show:
        j = 1
        for octave in octaves:
            for blurred_image in octaves[octave]:
                plt.subplot(nb_octave, pictures, j)
                plt.imshow(blurred_image, cmap="gray")
                j += 1

        plt.show()
    return octaves


octaves = scale_space(img)
diff_gaussian(octaves, show=True)
