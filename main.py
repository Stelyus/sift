import numpy as np
import sys
import matplotlib.pyplot as plt
import scipy
import os
from keypoints import laplacian

# First get the descriptors and the image

def m(x,y):
   pass 



if __name__ == "__main__":
    path_paris = '/users/franckthang/work/personalwork/sift/resources/paris.jpg'
    path_cat = '/users/franckthang/work/personalwork/sift/resources/cat.jpg'
    img = np.array(image.open(path_cat).convert('l'))

    octaves, dog, kps = laplacian.run(img)
