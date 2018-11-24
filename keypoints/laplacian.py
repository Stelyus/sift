import numpy as np
from PIL import Image
from scipy import signal, ndimage
import  matplotlib.pyplot as plt
import sys
import itertools

'''
References:
    https://www.cs.ubc.ca/~lowe/papers/ijcv04.pdf
    http://homepages.inf.ed.ac.uk/rbf/AVINVERTED/STEREO/av5_siftf.pdf

The optimal position is (x,y,t) + opt_x where
opt_x = - H^-1 * dx_matrix

'''
def is_extrema(x, y, down, mid, up):
    # 26 comparaions to check if the point if a extrema (min or max)
    permuts = list(set(itertools.permutations([-1,-1,1,1,0,0], 2)))
    mymin =  mymax = mid[x,y]
    rootx = x
    rooty = y
    for xx, yy in permuts:
        x = rootx + xx
        y = rooty + yy

        # out of bounds

        if x < 0 or y < 0 or x >= mid.shape[0] or y >= mid.shape[1]:
            return False

        mymin = min(down[x,y], mid[x,y], up[x,y], mymin)
        mymax = max(down[x,y], mid[x,y], up[x,y], mymax)

    return mymin != mid[rootx,rooty] and mymax == mid[rootx,rooty]


def locate_minimum(diff_gaussian, dict_std):
    kps = {n: [] for n in diff_gaussian.keys()}

    for key in diff_gaussian:
        pictures = diff_gaussian[key]
        for idx in range(1, len(pictures) -1):
            pic = pictures[idx]
            h, w = pic.shape
            for i in range(h):
                for j in range(w):
                    # If it is a local minimum or maximum
                    if is_extrema(i,j, pictures[idx-1], pic,
                                        pictures[idx+1]):


                       value = pic[i,j]
                       sup_pic, sub_pic= pictures[idx+1], pictures[idx-1]

                       # Computing dx_matrix
                       dx = (pic[i,j+1] - pic[i,j-1]) * .5 / 255
                       dy = (pic[i+1,j] - pic[i-1,j]) * .5 / 255
                       dt = (sup_pic[i,j]- sub_pic[i,j]) * .5 / 255
                       dx_matrix = np.matrix([[dx],[dy],[dt]])

                       # Computing Hessian matrix
                       dxx = (pic[i,j+1] + pic[i,j-1] - 2 * value) / 255
                       dyy = (pic[i+1,j] + pic[i-1,j] - 2 * value) / 255
                       dtt  = (sub_pic[i,j] + sub_pic[i,j] - 2 * value) / 255
                       dxy = (pic[i+1,j+1] - pic[i+1,j-1] - pic[i-1,j+1] + pic[i-1,j-1]) * 0.25  / 255
                       dxt = (sup_pic[i,j+1] - sup_pic[i,j-1] - sub_pic[i,j+1] + sub_pic[i,j-1])* 0.25 / 255
                       dyt = (sup_pic[i+1,j] - sup_pic[i-1,j] - sub_pic[i+1,j] + sub_pic[i-1,j]) * 0.25 / 255
                       hessian_matrix = np.matrix([[dxx,dxy,dxt],[dxy,dyy,dyt], [dxt,dyt,dtt]])

                       # Predict DoG value at subpixel extrema
                       try:
                           opt_X = -np.linalg.inv(hessian_matrix) @ dx_matrix
                       except:
                           continue

                       # Low contrast extrema prunning
                       p = np.absolute(value + .5 * (dx_matrix.T @ opt_X))
                       detH2 = (dxx * dyy) - (dxy ** 2)
                       traceH2 = dxx + dyy
                       if p < .03 or detH2 <= 0 \
                           or (traceH2 ** 2) / detH2 > 12 \
                           or np.count_nonzero(opt_X < .5)  != 3:
                           continue

                       kps[key].append((i,j))
    return kps
             
# Show contours
def diff_gaussian(octaves, show=False):
    diff_gaussian = {n: [] for n in octaves}
    for key in octaves:
        pictures = octaves[key]
        for idx in range(1, len(pictures)):
           pic1 = pictures[idx].astype('float64')
           pic2 = pictures[idx-1].astype('float64')
           pic_gauss = (pic1 - pic2)
           diff_gaussian[key].append(pic_gauss)

    if show:
        j = 1
        for key in diff_gaussian:
            for picture in diff_gaussian[key]:
                plt.subplot(len(diff_gaussian), len(diff_gaussian[key]), j)
                plt.imshow(picture, cmap="gray")
                j += 1
        plt.show()
    return diff_gaussian


def scale_space(img, show=False):
    '''
        s: number of pictures
        k: constant factor for each adjacents scales
    '''
    s = 5
    nb_octave = 4
    k = np.power(2, 1/(s-1))
    std = np.sqrt(.5)

    # Here for each octave we have the same std
    octaves = {n: [] for n in range(1,nb_octave+1)}
    dict_std = {n: [] for n in range(1,nb_octave+1)} 
    image = img

    for octave in range(1, nb_octave + 1):
        for i in range(s):
            new_std = std * np.power(k, i)
            dict_std[octave].append(new_std)
            blurred = ndimage.filters.gaussian_filter(image, new_std)
            octaves[octave].append(blurred)
        image = Image.fromarray(image)
        image = image.resize((image.size[0]//2,image.size[1]//2))
        image = np.array(image) 
    if show:
        for octave in octaves:
            j = 1
            for blurred_image in octaves[octaves]:
                plt.subplot(1, s, j)
                plt.imshow(blurred_image, cmap="gray")
                j += 1
            plt.show()

    return octaves, dict_std


def run(img):
    octaves, dict_std = scale_space(img)
    dog = diff_gaussian(octaves,show=True)
    kps = locate_minimum(dog, dict_std)
    return octaves, dog, kps