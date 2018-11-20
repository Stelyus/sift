import numpy as np
from PIL import Image
from scipy import signal
import  matplotlib.pyplot as plt
import sys



def laplacian_keypoints(img):

    gaussian_matrix = np.array([[1,4,6,4,1],\
                             [4,16,24,16,4],\
                             [6,24,36,24,6],\
                             [4,16,24,16,4],\
                             [1,4,6,4,1]])

    gaussian_matrix = gaussian_matrix / 256

    print(gaussian_matrix)
    blurred = img
    for _ in range(50):
        blurred = signal.convolve2d(blurred, gaussian_matrix, 'same').astype('uint8')
        print(blurred)


    #blurred = np.expand_dims(blurred, axis=-1)
    fig = plt.figure()
    ax = fig.add_subplot(121)
    ax1 = fig.add_subplot(122)
    ax.imshow(img, cmap="gray")
    ax.set_title("Original image")

    ax1.imshow(blurred, cmap="gray")
    ax1.set_title("Blurred image")
    plt.show()

path = '/Users/franckthang/Work/PersonalWork/sift/resources/cat.jpg'
img = np.array(Image.open(path).convert('L'))
print("Image shape: {!r}".format(img.shape))

laplacian_keypoints(img)


