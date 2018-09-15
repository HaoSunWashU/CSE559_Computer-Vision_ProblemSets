## Default modules imported. Import more if you need to.

import numpy as np
import scipy
from skimage.io import imread, imsave
import math

# Fill this out
# X is input color image
# K is the support of the filter (2K+1)x(2K+1)
# sgm_s is std of spatial gaussian
# sgm_i is std of intensity gaussian
def bfilt(X,K,sgm_s,sgm_i): #sgm_s and sgm_i are parameters for B[n1,n2]
    # Placeholder
    width = X.shape[0]
    height = X.shape[1]
    newX = np.zeros(X.shape) #也是有三个channel的
    sgm_s2 = 2*sgm_s*sgm_s
    sgm_i2 = 2*sgm_i*sgm_i
    #计算距离矩阵
    distanceMatrix = np.zeros((2*K + 1, 2*K + 1)) #是固定的，所以在外面算，提高效率
    for x in range(2*K + 1):
        for y in range(2*K + 1):
            distanceMatrix[x,y] = (math.pow(x-K,2) + math.pow(y-K,2))
    distanceMatrix = np.exp(-distanceMatrix/sgm_s2)

    for channel in range(3): #分为三个channel
        for i in range(width):
            for j in range(height):#扫描第n个channel的(i,j)点，对于(i,j)点 要循环求出对应这个点的B B的坐标从(0,0)到(2k,2k),中心点为(k,k)
                #开始B的循环,就是求B
                #print(i,j,channel)
                B = np.zeros((2 * K + 1, 2 * K + 1))
                neighborOfX = np.zeros((2*K+1,2*K+1)) #建立一个小X以(i,j)为中心，从X中获取值，然后和B内积求和
                for x in range(2*K+1):
                    for y in range(2*K+1):
                        #对于B中的(x,y)点，对应X中的(i-K+x, j-K+y)
                        if (i-K+x < 0) or (i-K+x > width-1) or (j-K+y < 0 ) or (j-K+y > height-1):
                            B[x,y] = 0
                            neighborOfX[x,y] = 0
                        else:
                            #distance = (math.pow(x-K,2) + math.pow(y-K,2))
                            intensity = (math.pow(X[i-K+x,j-K+y,channel] - X[i,j,channel], 2))
                            neighborOfX[x, y] = X[i - K + x, j - K + y, channel]
                            B[x,y] = math.exp(-intensity/(sgm_i2))
                B = B*distanceMatrix
                sumOfB = np.sum(B)
                B = B/sumOfB #归一化
                newX[i,j,channel] = np.sum(B*neighborOfX) #内积后求和
    #print(newX)

    return newX


########################## Support code below

def clip(im):
    return np.maximum(0.,np.minimum(1.,im))

from os.path import normpath as fn # Fixes window/linux path conventions
import warnings
warnings.filterwarnings('ignore')

img1 = np.float32(imread(fn('inputs/p4_nz1.png')))/255.
img2 = np.float32(imread(fn('inputs/p4_nz2.png')))/255.
print(img1.shape)

K=9

# print("Creating outputs/prob4_1_a.png")
# im1A = bfilt(img1,K,2,0.5)
# imsave(fn('outputs/prob4_1_a.png'),clip(im1A))


# print("Creating outputs/prob4_1_b.png")
# im1B = bfilt(img1,K,4,0.25)
# imsave(fn('outputs/prob4_1_b.png'),clip(im1B))

# print("Creating outputs/prob4_1_c.png")
# im1C = bfilt(img1,K,16,0.125)
# imsave(fn('outputs/prob4_1_c.png'),clip(im1C))
#
# # Repeated application
# print("Creating outputs/prob4_1_rep.png")
# im1D = bfilt(img1,K,2,0.125)
# for i in range(8):
#     im1D = bfilt(im1D,K,2,0.125)
# imsave(fn('outputs/prob4_1_rep.png'),clip(im1D))
#
# # Try this on image with more noise
print("Creating outputs/prob4_2_rep.png")
im2D = bfilt(img2,9,8,0.125)
for i in range(12):
    im2D = bfilt(im2D,K,2,0.125)
imsave(fn('outputs/prob4_2_rep.png'),clip(im2D))
