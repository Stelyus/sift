import numpy as np
import sys
import matplotlib.pyplot as plt
import scipy
import os
from keypoints import laplacian
from PIL import Image

def m(i,j, picture):
    ft = (picture[i,j+1] - picture[i,j-1]) ** 2
    st = (picture[i+1,j] - picture[i-1,j]) ** 2
    return np.sqrt(ft + st)

def theta(i,j,picture):
    ft = picture[i+1,j] - picture[i-1,j]
    st = picture[i,j+1] - picture[i,j-1]
    return np.arctanh(ft / st)

if __name__ == "__main__":
    path_paris = '/Users/franckthang/Work/Personalwork/sift/resources/paris.jpg'
    path_cat = '/Users/franckthang/Work/Personalwork/sift/resources/cat.jpg'
    img = np.array(Image.open(path_cat).convert('L'))

    infos =  laplacian.run(img) 
