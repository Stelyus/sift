import numpy as np
import sys
import matplotlib.pyplot as plt
import scipy
import os
import pickle

from keypoints import laplacian
from  PIL import Image


'''
References:
    http://aishack.in/tutorials/sift-scale-invariant-feature-transform-keypoint-orientation/
'''

'''
    infos: dict_keys([1, 2, 3, 4])
    infos[1]: dict_keys(['laplacian', 'dog', 'kps'])
    
    infos[1]["laplacian"] = [img0, img1...]
    infos[1]["kps"] = [tuple, tuple1...]
'''


'''
    The size of the "orientation collection region" around the keypoint depends on
    it's scale. The bigger the scale, the bigger the collection region.
'''

def is_valid_position(i,j,m_i,m_j):
    return not (i < 0 or j < 0 or i >= m_i or j >= m_j)


### Compute magnitude
def m(i,j, picture):
    if not is_valid_position(i,j,picture.shape[0], picture.shape[1]):
        return None
        
    ft = (picture[i,j+1] - picture[i,j-1]) ** 2
    st = (picture[i+1,j] - picture[i-1,j]) ** 2
    return np.sqrt(ft + st)

### Compute orientation
def theta(i,j,picture,local_m):
    if not is_valid_position(i,j,picture.shape[0], picture.shape[1]):
        return None
    
    ft = picture[i+1,j] - picture[i-1,j]
    st = picture[i,j+1] - picture[i,j-1]
    quotient = ft / st
 
    # ft=-17.0,st=0
    return np.arctanh(quotient)
    

def assign_orientation(infos):
    for octave in infos.keys():
        laplacian = infos[octave]['laplacian']
        kps = infos[octave]['kps']
        picture = laplacian[0].astype("float64")
        for i, j in kps:
            local_m = m(i,j,picture)
            local_t = theta(i,j,picture,local_m)
            print("Orientations: {}, Magnitude: {}\n".format(local_t, local_m))
            sys.exit()
            
    return infos


### Showing keypoints for the first octave's picture
def show_keypoints(infos):
    pic = infos[1]['laplacian'][0]
    for x,y in infos[1]['kps']:
        pic[x,y] = 255
    plt.imshow(pic, cmap="gray")
    plt.show()

if __name__ == "__main__":
    path_paris = '/Users/franckthang/Work/PersonalWork/sift/resources/paris.jpg'
    path_cat = '/Users/franckthang/Work/PersonalWork/sift/resources/cat.jpg'
    img = np.array(Image.open(path_paris).convert('L'))
    infos = pickle.load(open("pickle/infos_cat.pickle", "rb"))
    assign_orientation(infos)
    
    #infos = pickle.load(open("pickle/infos_paris.pickle", "rb"))
    #show_keypoints(infos)
    
    #ndimage.filters.gaussian_filter(image, new_std)

    '''
        infos =  laplacian.run(img) 
        pickle.dump(infos, open("infos_paris.pickle", "wb"))
    '''
