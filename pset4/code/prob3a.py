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

# Compute a 5x5 census transform of the grayscale image img.
# Return a uint32 array of the same shape
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


# Copy this from solution to problem 2.
def buildcv(left,right,dmax):
    imgC_left = census(left)
    imgC_right = census(right)
    # the third dimension (0-Dmax) all are 24, initially.
    cv = 24 * np.ones([left.shape[0], left.shape[1], dmax + 1], dtype=np.float32)
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

            for k in range(D + 1):
                cv[i, j, k] = hamdistArray[0, D - k]

                # # use hamdistArray to update cv's third dimension
                # # print(cv[i,j,0:D].shape)
                # cv[i, j, 0:D] = hamdistArray[0, 0:D]
    return cv

#preference function for d1 and d2
def Sfunc(d1,d2,P1,P2):
    if d1 == d2:
        return 0
    if abs(d1-d2) == 1:
        return P1
    return P2

# Implement the forward-backward viterbi method to smooth
# only along horizontal lines. Assume smoothness cost of
# 0 if disparities equal, P1 if disparity difference <= 1, P2 otherwise.
#
# Function takes in cost volume cv, and values of P1 and P2
# Return the disparity map
def viterbilr(cv,P1,P2):
    H = cv.shape[0]
    W = cv.shape[1]
    D = cv.shape[2]
    C_ = cv
    disMap = np.zeros((H,W), dtype=np.float32) #disparity map
    #construct preference matrix for two d
    S = np.zeros((D,D))
    for i in range(D):
        for j in range(D):
            S[i,j] = Sfunc(i,j,P1,P2)

    print(S)
    # construct a Z for recording different chains
    Z = np.zeros(cv.shape)

    for y in range(H):
        #****    forward    *****
        #the first node
        C_[y,0,:] = cv[y,0,:]

        for x in range(1,W):
            #compute d' for Z[y,x,d] & compute Z_[y,x,:]
            for d in range(0,D):
                Z[y,x,d] = np.argmin(S[:,d] + C_[y,x-1,:])
                # d_ = Z[y,x,d]
                C_[y,x,d] = cv[y,x,d] + S[int(Z[y,x,d]),d] + C_[y,x-1,int(Z[y,x,d])]
                # C_[y,x,d] = cv[y,x,d] + np.min(S[:,d] + C_[y,x-1,:])
        #compute the last node in one row
        disMap[y,W-1] = np.argmin(C_[y,W-1,:])
        #***    backward    ****
        for x in range(0,W-1):
            x_ = W-2-x
            disMap[y,x_] = Z[y,x_+1,int(disMap[y,x_+1])]

        # print(Z[y,:,:])

    #return disparity computed by forward and backward
    return disMap
    # return np.argmin(cv,axis=2)
    
    
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
d = viterbilr(cv,0.5,16)
print(d)
# Map to color and save
dimg = cm.jet(np.minimum(1,np.float32(d.flatten())/50.))[:,0:3]
dimg = dimg.reshape([d.shape[0],d.shape[1],3])
imsave(fn('outputs/prob3a.jpg'),dimg)
