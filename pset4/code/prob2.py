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
    img_bigger = np.zeros((H+4, W+4))
    img_bigger[2:H+2,2:W+2] = img  # a bigger img with 0 out side the bound of img, it is for get 5*5 neighbor array
    c = np.zeros([H,W],dtype=np.uint32) # unsigned int 32bits, only use lower 24bits
    for i in range(H):
        for j in range(W):
            #5*5 array with all the same value img[i,j]
            X = np.zeros((5,5))
            X = img[i,j]

            #5*5 array with the center img[i,j] and its neighbor in img. if neighbor out of bound of img, set value to zero
            neighbor = np.zeros((5,5))
            i_ = i + 2
            j_ = j + 2
            neighbor = img_bigger[i_-2:i_+3, j_-2:j_+3]

            result = np.zeros((5,5))
            result = X - neighbor
            result = np.where(result > 0, 1 , 0) # img[i,j] > neighbor

            sum = 0
            for x in range(5):
                for y in range(5):
                    if(x*5+y <= 11): # first 12 bits
                        sum = sum + 2**(x*5 + y) * result[x,y]
                    if(x*5+y >= 13):
                        sum = sum + 2**(x*5 + y - 1) * result[x,y]
            c[i,j] = sum
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


# Fill this out
# CV is the cost-volume to be filtered.
# X is the left color image that will serve as guidance.
# K is the support of the filter (2K+1)x(2K+1)
# sgm_s is std of spatial gaussian
# sgm_i is std of intensity gaussian
#
# Feel free to look at the solution key for bfilt function form problem set 1.
def bfilt(cv,X,K,sgm_s,sgm_i):
    H = X.shape[0]
    W = X.shape[1]

    yy = np.zeros(cv.shape) # same shape with cv
    B = np.zeros([H,W,1])

    for y in range (-K, K+1):
        for x in range (-K, K+1):
            if y < 0:
                y1a = 0
                y1b = -y
                y2a = H + y
                y2b = H
            else:
                y1a = y
                y1b = 0
                y2a = H
                y2b = H - y

            if x < 0:
                x1a = 0
                x1b = -x
                x2a = W + x
                x2b = W
            else:
                x1a = x
                x1b = 0
                x2a = W
                x2b = W-x

            #compute weight kernel based on X
            bxy = X[y1a:y2a, x1a:x2a, :] - X[y1b:y2b, x1b:x2b, :]
            bxy = np.sum(bxy*bxy, axis=2, keepdims=True)

            bxy = bxy/(sgm_i**2) + np.float32(y**2 + x**2) / (sgm_s ** 2)
            bxy = np.exp(-bxy/2.0)

            B[y1b:y2b, x1b:x2b, :] = B[y1b:y2b, x1b:x2b, :] + bxy
            #apply to cost volume C
            yy[y1b:y2b, x1b:x2b, :] = yy[y1b:y2b, x1b:x2b, :] + bxy*cv[y1a:y2a, x1a:x2a, :]

    return yy/B

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

cv0 = buildcv(left_g,right_g,50) # cv0 is the cost-volume, cv0 is H*W*(dmax+1)

cv1 = bfilt(cv0,left,5,2,0.5)


d0 = np.argmin(cv0,axis=2)
d1 = np.argmin(cv1,axis=2)

# Map to color and save
dimg = cm.jet(np.minimum(1,np.float32(d0.flatten())/50.))[:,0:3]
dimg = dimg.reshape([d0.shape[0],d0.shape[1],3])
imsave(fn('outputs/prob2a.jpg'),dimg)

dimg = cm.jet(np.minimum(1,np.float32(d1.flatten())/50.))[:,0:3]
dimg = dimg.reshape([d1.shape[0],d1.shape[1],3])
imsave(fn('outputs/prob2b.jpg'),dimg)
