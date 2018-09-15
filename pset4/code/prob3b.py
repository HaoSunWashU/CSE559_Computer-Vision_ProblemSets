## Default modules imported. Import more if you need to.

import numpy as np


#########################################
### Hamming distance computation
### You can call the function hamdist with two
### uint32 bit arrays of the same size. It will
### return another array of the same size with
### the elmenet-wise hamming distance.
hd8bit = np.zeros((256,))
for i in range(256):
    v = i
    for k in range(8):
        hd8bit[i] = hd8bit[i] + v%2
        v=v//2


def hamdist(x,y):
    dist = np.zeros(x.shape)
    g = x^y
    for i in range(4):
        dist = dist + hd8bit[g%256]
        g = g // 256
    return dist
#########################################

#preference function for d1 and d2
def Sfunc(d1,d2,P1,P2):
    if d1 == d2:
        return 0
    if abs(d1-d2) == 1:
        return P1
    return P2

# Compute a 5x5 census transform of the grayscale image img.
# Return a uint32 array of the same shape
# def census(img):
#     # img = np.int16(img0)
#     H = img.shape[0]
#     W = img.shape[1]
#     img_bigger = np.zeros((H+4, W+4))
#     img_bigger[2:H+2,2:W+2] = img  # a bigger img with 0 out side the bound of img, it is for get 5*5 neighbor array
#     c = np.zeros([H,W],dtype=np.uint32) # unsigned int 32bits, only use lower 24bits
#     for i in range(H):
#         for j in range(W):
#             #5*5 array with all the same value img[i,j]
#             X = np.zeros((5,5))
#             X = img[i,j]
#
#             #5*5 array with the center img[i,j] and its neighbor in img. if neighbor out of bound of img, set value to zero
#             neighbor = np.zeros((5,5))
#             i_ = i + 2
#             j_ = j + 2
#             neighbor = img_bigger[i_-2:i_+3, j_-2:j_+3]
#
#             result = np.zeros((5,5))
#             result = X - neighbor
#             result = np.where(result > 0, 1 , 0) # img[i,j] > neighbor
#
#             sum = 0
#             for x in range(5):
#                 for y in range(5):
#                     if(x*5+y <= 11): # first 12 bits
#                         sum = sum + 2**(x*5 + y) * result[x,y]
#                     if(x*5+y >= 13):
#                         sum = sum + 2**(x*5 + y - 1) * result[x,y]
#             c[i,j] = sum
#     return c

def census(img):
    # img = np.int16(img0)
    H = img.shape[0]
    W = img.shape[1]
    c = np.zeros([H,W], dtype = np.uint32)
    inc = np.uint32(1)
    for dx in range(-2,3):
        for dy in range(-2,3):
            if dx == 0 and dy == 0:
                continue

            cx0 = np.maximum(0, -dx); dx0 = np.maximum(0,dx)
            cx1 = W-dx0; dx1 = W-cx0
            cy0 = np.maximum(0,-dy); dy0 = np.maximum(0,dy)
            cy1 = H-dy0; dy1 = H-cy0

            c[cy0:cy1, cx0:cx1] = c[cy0:cy1, cx0:cx1] + \
                inc*(img[cy0:cy1, cx0:cx1] > img[dy0:dy1, dx0:dx1])
            inc = inc * 2

    return c

# Given left and right grayscale images and max disparity D_max, build a HxWx(D_max+1) array
# corresponding to the cost volume. For disparity d where x-d < 0, fill a cost
# value of 24 (the maximum possible hamming distance).
#
# You can call the hamdist function above, and copy your census function from the
# previous problem set.
def buildcv(left,right,dmax):
    imgC_left = census(left)
    imgC_right = census(right)
    # the third dimension (0-Dmax) all are 24, initially.
    cv = 24 * np.ones([left.shape[0],left.shape[1],dmax+1], dtype=np.float32)
    H = left.shape[0]
    W = left.shape[1]

    # j is x, i is y
    for i in range(H):  # y
        for j in range(W):  # x
            D = dmax  # D is the current dmax
            if (j - dmax < 0):
                D = j
            pointsC_left = np.zeros((1, D + 1), dtype=np.uint32)
            pointsC_right = np.zeros((1, D + 1), dtype=np.uint32)
            pointsC_left[0, :] = imgC_left[i, j]
            pointsC_right[0, :] = imgC_right[i, j - D:j + 1]
            hamdistArray = hamdist(pointsC_right, pointsC_left)  # hamdistArray 1 * D

            for k in range(D+1):
                cv[i, j, k] = hamdistArray[0, D-k]

            # # use hamdistArray to update cv's third dimension
            # # print(cv[i,j,0:D].shape)
            # cv[i, j, 0:D] = hamdistArray[0, 0:D]
    return cv


# Do SGM. First compute the augmented / smoothed cost volumes along 4
# directions (LR, RL, UD, DU), and then compute the disparity map as
# the argmin of the sum of these cost volumes. 
def SGM(cv,P1,P2):
    H = cv.shape[0]
    W = cv.shape[1]
    D = cv.shape[2]
    C_lr = cv
    C_rl = cv
    C_du = cv
    C_ud = cv

    disMap = np.zeros((H, W), dtype=np.int32)  # disparity map
    # construct preference matrix for two d
    S = np.zeros((D, D))
    for i in range(D):
        for j in range(D):
            S[i, j] = Sfunc(i, j, P1, P2)

    #Horizontal
    for y in range(H):
        # ****    forward    *****
        # the first node
        C_lr[y, 0, :] = cv[y, 0, :]
        C_rl[y, W - 1, :] = cv[y, W - 1, :]
        for x in range(1, W):
            x_ = W - 1 - x
            for d in range(0, D):
                # from Left to Right
                C_lr[y, x, d] = cv[y, x, d] + np.min(S[:, d] + C_lr[y, x - 1, :])
                # from Right to Left
                C_rl[y, x_, d] = cv[y, x_, d] + np.min(S[:, d] + C_rl[y, x_ + 1, :])

    #vertical
    for x in range(W):
        #****    forward    *****
        #the first node
        C_ud[0, x, :] = cv[0, x, :]
        C_du[H - 1, x, :] = cv[H - 1, x, :]
        for y in range(1, H):
            y_ = H - 1 - y
            for d in range(0, D):
                # from up to down
                C_ud[y, x, d] = cv[y, x, d] + np.min(S[:, d] + C_ud[y - 1, x, :])
                # from down to up
                C_du[y_, x, d] = cv[y_, x, d] + np.min(S[:, d] + C_du[y_ + 1, x, :])

    # compute disparity map
    disMap = np.argmin((C_lr + C_rl + C_ud + C_du), axis = 2)

    # return disparity
    return disMap

########################## Support code below

from skimage.io import imread, imsave
from os.path import normpath as fn # Fixes window/linux path conventions
import matplotlib.cm as cm
import warnings
warnings.filterwarnings('ignore')


left = np.float32(imread(fn('inputs/left.jpg')))/255.
right = np.float32(imread(fn('inputs/right.jpg')))/255.

left_g = np.mean(left,axis=2)
right_g = np.mean(right,axis=2)
                   
cv = buildcv(left_g,right_g,50)
d = SGM(cv,0.5,16)

# Map to color and save
dimg = cm.jet(np.minimum(1,np.float32(d.flatten())/50.))[:,0:3]
dimg = dimg.reshape([d.shape[0],d.shape[1],3])
imsave(fn('outputs/prob3b.jpg'),dimg)
