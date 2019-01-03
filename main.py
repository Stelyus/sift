#!/anaconda3/bin/python

import os
import pickle
import itertools
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image
from keypoints import laplacian
from scipy.stats import multivariate_normal

'''
References:
    http://aishack.in/tutorials/sift-scale-invariant-feature-transform-keypoint-orientation/
    http://www.vlfeat.org/api/sift.html
'''

'''
    infos: dict_keys([1, 2, 3, 4])
    infos[1]: dict_keys(['original', 'gaussian', 'std' , 'dog', 'kps'])
    
    infos[1]["gaussian"] = [img0, img1...]
    infos[1]["kps"] = [tuple, tuple1...]
'''


'''
    The size of the "orientation collection region" around the keypoint depends
    on
    it's scale. The bigger the scale, the bigger the collection region.
'''

### Compute gradient magnitude
def gradient_m(i,j, picture):
    ft = (picture[i,j+1] - picture[i,j-1]) ** 2
    st = (picture[i+1,j] - picture[i-1,j]) ** 2

    return np.sqrt(ft + st)

### Compute gradient orientation
def gradient_theta(i,j,picture):
    #To avoid error division by zero
    eps = 1e-5
    
    ft = picture[i+1,j] - picture[i-1,j]
    st = (picture[i,j+1] - picture[i,j-1]) + eps

    ret = 180 + np.arctan2(ft, st)  * 180 / np.pi
    return ret


def dominant_hist(hist):
    sorted_hist = np.argsort(hist, axis=0)[::-1]
    max_hist = hist[sorted_hist[0]]
    i = 1
    while .8 * max_hist < hist[sorted_hist[i]]:
        print("Dominant i arg {}".format(i))
        i += 1


def create_histogram(i,j,picture,std):
    truncate = 4.0
    kernel_size = 2 * int(std * truncate + .5) + 1
    window  = list(range(-kernel_size, kernel_size + 1))

    diag  = set(itertools.permutations(window, 2))
    rooti, rootj = i,j
    theta_list =  []

    gaussian = multivariate_normal(mean=[i,j], cov=1.5*std)
    orient_hist = np.zeros([36,1])
    
    for ii, jj in diag:
        x = rooti + ii
        y = rootj + jj
        if x - 1 < 0 or y - 1 < 0 or x + 1 > picture.shape[0] - 1 \
            or y + 1 > picture.shape[1] -1:
            continue
        
        # TODO: Warning the magnitude are really small
        magnitude = gradient_m(x,y,picture)
        weight =  magnitude * gaussian.pdf([x,y])

        orientation = gradient_theta(x,y,picture)
        bins_orientation = np.clip(orientation // 10, 0,35)

        orient_hist[int(bins_orientation)] += weight

    return orient_hist


def assign_orientation(infos):
    index = 0 
    for octave in infos.keys():
        kps = infos[octave]['kps']
        std = infos[octave]["std"][index]
        picture = infos[octave]['gaussian'][index].astype('float64')

        for i, j in kps: 
            hist = create_histogram(i,j,picture,std)
            dominant_hist(hist)
        print("Next octave")
    return infos

### Showing keypoints for the first octave's picture
def show_keypoints(infos):
    #pic = np.zeros(infos[1]['gaussian'][0].shape)
    pic = infos[1]['gaussian'][0]
    for x,y in infos[1]['kps']:
        pic[x,y] = 255
    
    plt.imshow(pic, cmap="gray")
    plt.show()

def run(load=None, img=None):
    infos = None
    if load  is not None:
        infos = pickle.load(open(load, "rb"))
        assign_orientation(infos)
        #show_keypoints(infos)
    if img is not None:
        infos = laplacian.run(img) 
    return infos 

def reload():
    def step(path, name):
        img = np.array(Image.open(path).convert('L'))
        infos = run(img=img)
        pickle.dump(infos, open("pickle/infos_{}.pickle".format(name), "wb"))
        
    path_paris = '/Users/franckthang/work/PersonalWork/sift/resources/paris.jpg'
    path_cat = '/Users/franckthang/work/PersonalWork/sift/resources/cat.jpg'
    
    step(path_paris, "paris")
    step(path_cat, "cat")

if __name__ == "__main__":
    path_paris = '/Users/franckthang/work/PersonalWork/sift/resources/paris.jpg'
    path_cat = '/Users/franckthang/work/PersonalWork/sift/resources/cat.jpg'
    img = np.array(Image.open(path_cat).convert('L'))
    run(load="pickle/infos_paris.pickle")

    #reload()
