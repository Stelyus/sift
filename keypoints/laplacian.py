import numpy as np
from PIL import Image
from scipy import signal, ndimage, misc
import  matplotlib.pyplot as plt
import sys
import itertools

'''
References:
    https://www.cs.ubc.ca/~lowe/papers/ijcv04.pdf
    http://homepages.inf.ed.ac.uk/rbf/AVINVERTED/STEREO/av5_siftf.pdf
'''

DIAG = list(set(itertools.permutations([-1,-1,1,1,0,0], 2)))

def is_extrema(x, y, down, mid, up):
    # 26 comparaisons to check if the point is a extrema (min or max)
    mymin = mymax = mid[x,y]
    rootx, rooty = x, y
    for xx, yy in DIAG:
        x = rootx + xx
        y = rooty + yy

        # out of bounds
        if x < 0 or y < 0 or x >= mid.shape[0] or y >= mid.shape[1]:
            return False

        mymin = min(down[x,y], mid[x,y], up[x,y], mymin)
        mymax = max(down[x,y], mid[x,y], up[x,y], mymax)

    # check if x is still the max or min
    return mymin != mid[rootx,rooty] and mymax == mid[rootx,rooty]


def locate_minimum(infos):
    for key in infos:
        infos[key]["kps"] = []
        pictures = infos[key]["dog"]
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
                       
                       h1 = [dxx,dxy,dxt]
                       h2 = [dxy,dyy,dyt]
                       h3 = [dxt,dyt,dtt]
                       hessian_matrix = np.matrix([h1,h2,h3])

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

                       #TODO: Should append the amount of blur ?
                       infos[key]["kps"].append((i,j))
             
# Show contours
def diff_gaussian(infos, show=False):
    for key in infos:
        infos[key]["dog"] = []
        pictures = infos[key]["laplacian"]
        for idx in range(1, len(pictures)):
           pic1 = pictures[idx].astype('float64')
           pic2 = pictures[idx-1].astype('float64')
           pic_gauss = (pic1 - pic2)
           infos[key]["dog"].append(pic_gauss)

    if show:
        j = 1
        for key in infos:
            for picture in infos[key]["dog"]:
                plt.subplot(len(infos), len(infos[key]["dog"]), j)
                plt.imshow(picture, cmap="gray")
                j += 1
        plt.show()


# Creating linear scale space
def scale_space(img, infos, show=False):
    '''
        s: number of pictures
        k: constant factor for each adjacents scales
    '''
    
    s = 5
    nb_octave = 4
    k = np.power(2, 1/(s-1))
    std = np.sqrt(.5)
    # The first picture is a upsampling
    image = misc.imresize(img, 200, 'bilinear')
    for octave in range(1, nb_octave + 1):
        infos[octave] = {"laplacian": [], "std": []}
        infos[octave]['original'] = image
        for i in range(s):
            new_std = std * np.power(k,i)
            blurred = ndimage.filters.gaussian_filter(image, new_std)
            infos[octave]["std"].append(new_std)
            # Not really laplacian ... it's gaussian actually
            infos[octave]["laplacian"].append(blurred)
         
        # Updating std
        std = std * np.power(k, 2) 
        # Resizing the picture
        image = misc.imresize(image, 50, 'bilinear') 
    if show:
        for key in infos:
            j = 1
            for blurred_image in infos[key]["laplacian"]:
                plt.subplot(1, s, j)
                plt.imshow(blurred_image, cmap="gray")
                j += 1
            plt.show()
    exit(0)


def run(img):
    infos = {}
    scale_space(img, infos)
    diff_gaussian(infos)
    locate_minimum(infos)
    return infos
