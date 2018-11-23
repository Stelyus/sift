import numpy as np
from PIL import Image
from scipy import signal
import  matplotlib.pyplot as plt
import sys
import itertools

#path = '/Users/franckthang/Work/PersonalWork/sift/resources/paris.jpg'
path = '/Users/franckthang/Work/PersonalWork/sift/resources/cat.jpg'


img = np.array(Image.open(path).convert('L'))
print("Image initial shape: {}".format(img.shape))

def locate_minimum(diff_gaussian, dict_std):
    def is_extrema(x, y, down, mid, up):
        permuts = list(set(itertools.permutations([-1,-1,1,1,0,0], 2)))
        mymin =  mymax = mid[x,y]
        for xx, yy in permuts:
            x += xx
            y += yy
            # out of bounds
            if x < 0 or y < 0 or x >= mid.shape[0] or y >= mid.shape[1]:
                return False
            mymin = min(down[x,y], mid[x,y], up[x,y], mymin)
            mymax = max(down[x,y], mid[x,y], up[x,y], mymax)

        return mymin == mid[x,y] or mymax == mid[x,y]

    local_minmax = {n: [] for n in diff_gaussian.keys()}
    for key in diff_gaussian:
        pictures = diff_gaussian[key]
        it_pic = pictures[1:-1]
        for idx, picture in enumerate(it_pic):
            h, w = picture.shape
            for i in range(h):
                for j in range(w):
                    # If it is a local minimum or maximum
                    if is_extrema(i,j, pictures[idx-1], picture,
                                        pictures[idx+1]):
                       value = picture[i,j]
                       sup_pic, sub_pic= pictures[idx +1], pictures[idx-1]
                       x = np.array([i, j, dict_std[key][idx]])

                       # Computing dx_matrix

                       print(sup_pic[i,j], sub_pic[i,j], (sup_pic[i,j]- sub_pic[i,j]) / 2)
                       sys.exit()

                       dx = (picture[i,j+1] - picture[i,j-1]) / 2
                       dy = (picture[i+1,j] - picture[i-1,j]) / 2
                       dt = (sup_pic[i,j]- sub_pic[i,j]) / 2

                       dxx = picture[i,j+1] + picture[i,j-1] - 2 * value
                       dyy = picture[i+1,j] + picture[i-1,j] - 2 * value
                       dtt  = pictures[idx+1][i,j] + pictures[idx-1][i,j] - 2 * value
                       dxy = (picture[i+1,j+1] - picture[i+1,j-1] - picture[i-1,j+1] + picture[i-1,j-1]) * 0.25
                       dxt = (sup_pic[i,j+1] - sup_pic[i,j-1] - sub_pic[i,j+1] + sub_pic[i,j-1])* 0.25
                       dyt = (sup_pic[i+1,j] - sup_pic[i-1,j] - sub_pic[i+1,j] + sub_pic[i-1,j]) * 0,25

                       #x_hat = np.linalg.ltsq(hessian_matrix, dx_matrix)
                       # print(x_hat)
    return None
             
# Show contours
def diff_gaussian(octaves, show=False):
    diff_gaussian = {n: [] for n in octaves}
    for key in octaves:
        pictures = octaves[key]
        for idx, picture in enumerate(pictures[1:]):
           diff_gaussian[key].append(picture - pictures[idx-1])

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
    dict_std = {n: [] for n in range(1,nb_octave+1)}
    pictures = 5
    scale = [np.power(x,2) for x in range(1, nb_octave+1)][::-1]
    image = img
    std = np.sqrt(.5)

    for octave in range(1, nb_octave + 1):
        image = Image.fromarray(image)
        image = image.resize((image.size[0]//2,image.size[1]//2))
        image = np.array(image) 
        for i in range(pictures):
            new_std = std * np.power(np.sqrt(2), i)
            octaves[octave].append(blur(new_std, image))
            dict_std[octave].append(new_std)

    if show:
        j = 1
        for octave in octaves:
            for blurred_image in octaves[octave]:
                plt.subplot(nb_octave, pictures, j)
                plt.imshow(blurred_image, cmap="gray")
                j += 1

        plt.show()
    return octaves, dict_std

octaves, dict_std = scale_space(img)
dog = diff_gaussian(octaves)
locate_minimum(dog, dict_std)
#
#pic = np.zeros(dog[1][0].shape)
#pts = minimums[1]
#for x, y in pts:
    #pic[x,y] = 255
##
#plt.imshow(pic, cmap="gray")
#plt.show()
#
