## Default modules imported. Import more if you need to.

import numpy as np
from skimage.io import imread, imsave
from scipy.signal import convolve2d as conv2

# Different thresholds to try
from statsmodels.compat import scipy

T0 = 0.3
T1 = 0.5
T2 = 0.7


########### Fill in the functions below

# Return magnitude, theta of gradients of X
def grads(X):
    Dx = [[1,0,-1],
          [2,0,-2],
          [1,0,-1]]
    Dy = [[1,2,1],
          [0,0,0],
          [-1,-2,-1]]

    #placeholder
    H = np.zeros(X.shape,dtype=np.float32)
    theta = np.zeros(X.shape,dtype=np.float32)
    Ix = conv2(X, Dx, mode='same')
    Iy = conv2(X, Dy, mode='same')

    # print(X.shape)
    # print(Ix.shape)
    # print(Iy.shape)
    H = np.sqrt(np.square(Ix) + np.square(Iy))
    theta = np.arctan2(Iy,Ix)

    return H,theta

def nms(E,H,theta):
    #horizontal theta
    A1 = np.array([[0,0,0],[-1,1,0],[0,0,0]])
    A2 = np.array([[0,0,0],[0,1,-1],[0,0,0]])
    horizontal = np.logical_or(np.logical_or(np.logical_and(theta >= -np.pi/8, theta <= np.pi/8),
                                np.logical_and(theta <= np.pi, theta >= np.pi * 7/8)),
                               np.logical_and(theta >= -np.pi,theta <= -np.pi * 7/8)) # return array of true or false
    index = np.where(
        np.logical_and(horizontal, np.logical_and(conv2(H,A1, mode='same') <= 0,conv2(H,A2, mode='same') <= 0)))
    E[index] = 0

    # #vertical theta
    B1 = np.array([[0,-1,0],[0,1,0],[0,0,0]])
    B2 = np.array([[0,0,0],[0,1,0],[0,-1,0]])
    vertical = np.logical_or(np.logical_and(theta >= np.pi* 3/8, theta <= np.pi* 5/8),
                            np.logical_and(theta >= -np.pi * 5/8, theta <= -np.pi * 3/8)) # return array of true or false
    index = np.where(
        np.logical_and(horizontal, np.logical_and(conv2(H, B1, mode='same') <= 0, conv2(H, B2, mode='same') <= 0)))
    E[index] = 0

    # #right diagonal
    C1 = np.array([[0,0,-1],[0,1,0],[0,0,0]])
    C2 = np.array([[0, 0, 0], [0, 1, 0], [-1, 0, 0]])
    rightDia = np.logical_or(np.logical_and(theta >= np.pi/8, theta <= np.pi * 3/8),
                             np.logical_and(theta >= -np.pi * 7/8, theta <= -np.pi * 5/8))
    index = np.where(
        np.logical_and(horizontal, np.logical_and(conv2(H, C1, mode='same') <= 0, conv2(H, C2, mode='same') <= 0)))
    E[index] = 0

    # #left diagonal
    D1 = np.array([[-1,0,0],[0,1,0],[0,0,0]])
    D2 = np.array([[0, 0, 0], [0, 1, 0], [0, 0, -1]])
    leftDia = np.logical_or(np.logical_and(theta >= np.pi * 5/8, theta <= np.pi * 7/8),
                             np.logical_and(theta >= -np.pi * 3/8, theta <= -np.pi/8))
    index = np.where(
        np.logical_and(horizontal, np.logical_and(conv2(H, D1, mode='same') <= 0, conv2(H, D2, mode='same') <= 0)))
    E[index] = 0

    return E

########################## Support code below

from os.path import normpath as fn # Fixes window/linux path conventions
import warnings
warnings.filterwarnings('ignore')

img = np.float32(imread(fn('inputs/p3_inp.png')))/255.

H,theta = grads(img)
print(theta)

imsave(fn('outputs/prob3_a.png'),H/np.max(H[:]))

## Part b

E0 = np.float32(H > T0)  #binarized edge image
E1 = np.float32(H > T1)
E2 = np.float32(H > T2)

imsave(fn('outputs/prob3_b_0.png') , E0)
imsave(fn('outputs/prob3_b_1.png') , E1)
imsave(fn('outputs/prob3_b_2.png') , E2)

E0n = nms(E0,H,theta)
E1n = nms(E1,H,theta)
E2n = nms(E2,H,theta)

imsave(fn('outputs/prob3_b_nms0.png'),E0n)
imsave(fn('outputs/prob3_b_nms1.png'),E1n)
imsave(fn('outputs/prob3_b_nms2.png'),E2n)
