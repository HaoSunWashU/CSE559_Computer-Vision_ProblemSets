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
    dist = np.zeros(x.shape) #x, y are arrays of census value?

    g = x^y
    for i in range(4):
        dist = dist + hd8bit[g%256]
        g = g // 256

    return dist
#########################################

## Fill out these functions yourself

# Compute a 5x5 census transform of the grayscale image img.
# Return a uint32 array of the same shape
def census(img0):
    img = np.int16(img0)
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
    

# Given left and right image and max disparity D_max, return a disparity map
# based on matching with  hamming distance of census codes. Use the census function
# you wrote above.
#
# d[x,y] implies that left[x,y] matched best with right[x-d[x,y],y]. Disparity values
# should be between 0 and D_max (both inclusive).
def smatch(left,right,dmax):
    d = np.zeros(left.shape)
    imgC_left = census(left)
    # print(imgC_left)
    imgC_right = census(right)
    # print(imgC_right)
    H = left.shape[0]
    W = left.shape[1]

    # j is x, i is y
    for i in range(H):  # y
        for j in range(W): # x
            D = dmax # D is the current dmax
            if(j - dmax<0):
                D = j
            pointsC_left = np.zeros((1,D+1),dtype = np.uint32)
            pointsC_right = np.zeros((1,D+1),dtype = np.uint32)
            # print("pointsC_right")
            # print(pointsC_right.shape)
            # print("pointsC_left")
            # print(pointsC_left.shape)

            pointsC_left[0,:] = imgC_left[i,j]
            # print("pointsC_left")
            # print(pointsC_left.shape)

            pointsC_right[0,:] = imgC_right[i,j-D:j+1]
            # print("pointsC_right")
            # print(pointsC_right.shape)

            # hamdistArray = np.zeros(pointsC_right.shape)
            hamdistArray = hamdist(pointsC_right, pointsC_left) #hamdistArray 1 * D

            # print(hamdistArray)
            # print(hamdistArray.min())
            # print(hamdistArray.shape)

            minHamdist = hamdistArray.min()
            minIndexArray = np.where(hamdistArray==minHamdist)
            minIndexRow, minIndexColumn = minIndexArray
            minIndex = minIndexColumn[-1]
            # minIndex = minIndexArray[0] #minIndex (*,*) should get second index
            # a,b = minIndex
            d[i,j] = D - minIndex

    return d
    
    
########################## Support code below

from skimage.io import imread, imsave
from os.path import normpath as fn # Fixes window/linux path conventions
import matplotlib.cm as cm
import warnings
warnings.filterwarnings('ignore')


left = imread(fn('inputs/left.jpg'))
right = imread(fn('inputs/right.jpg'))

d = smatch(left,right,40)

# Map to color and save
dimg = cm.jet(np.minimum(1,np.float32(d.flatten())/20.))[:,0:3]
dimg = dimg.reshape([d.shape[0],d.shape[1],3])
imsave(fn('outputs/prob5.png'),dimg)
