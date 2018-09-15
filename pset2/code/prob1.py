## Default modules imported. Import more if you need to.

import numpy as np
from skimage.io import imread, imsave


## Fill out these functions yourself

# Copy from Pset1/Prob6 
def im2wv(img,nLev):
    # Placeholder that does nothing
    height = img.shape[0]
    width = img.shape[1]
    # print(width)
    # for x in range(0, width, 2):
    #     print(x)

    # create list
    list = []  # save different level's images
    list.append(img)
    for i in range(nLev):
        X = list[-1]  # get the last one of List
        del list[-1]  # remove the last one
        Xnew = np.zeros((int(height / pow(2, i + 1)), int(width / pow(2, i + 1))))
        H1 = np.zeros((int(height / pow(2, i + 1)), int(width / pow(2, i + 1))))
        H2 = np.zeros((int(height / pow(2, i + 1)), int(width / pow(2, i + 1))))
        H3 = np.zeros((int(height / pow(2, i + 1)), int(width / pow(2, i + 1))))
        for x in range(0, int(height / pow(2, i)), 2):
            for y in range(0, int(width / pow(2, i)), 2):
                # X[x,y]=a X[x,y+1]=b X[x+1,y]=c X[x+1,y+1]=d
                Xnew[int(x / 2), int(y / 2)] = (X[x, y] + X[x, y + 1] + X[x + 1, y] + X[x + 1, y + 1]) / 2
                H1[int(x / 2), int(y / 2)] = (X[x, y + 1] + X[x + 1, y + 1] - X[x, y] - X[x + 1, y]) / 2
                H2[int(x / 2), int(y / 2)] = (X[x + 1, y] + X[x + 1, y + 1] - X[x, y] - X[x, y + 1]) / 2
                H3[int(x / 2), int(y / 2)] = (X[x, y] + X[x + 1, y + 1] - X[x, y + 1] - X[x + 1, y]) / 2

        newList = [H1, H2, H3]
        list.append(newList)
        list.append(Xnew)  # put new size X into the last one

    return list


# Copy from Pset1/Prob6 
def wv2im(pyr):
    waveLetPyr = pyr
    length = len(waveLetPyr)
    lev = length - 1;
    for i in range(lev):
        L = waveLetPyr[-1]
        del waveLetPyr[-1]  # pyr.remove last one
        Hlist = waveLetPyr[-1]
        del waveLetPyr[-1]
        H1 = Hlist[0]
        H2 = Hlist[1]
        H3 = Hlist[2]
        height = L.shape[0]
        width = L.shape[1]
        Xnew = np.zeros((height * 2, width * 2))
        for x in range(height):
            for y in range(width):
                Xnew[2 * x, 2 * y] = (L[x, y] - H1[x, y] - H2[x, y] + H3[x, y]) / 2
                Xnew[2 * x, 2 * y + 1] = (L[x, y] + H1[x, y] - H2[x, y] - H3[x, y]) / 2
                Xnew[2 * x + 1, 2 * y] = (L[x, y] - H1[x, y] + H2[x, y] - H3[x, y]) / 2
                Xnew[2 * x + 1, 2 * y + 1] = (L[x, y] + H1[x, y] + H2[x, y] - H3[x, y]) / 2
        waveLetPyr.append(Xnew)

    # # Placeholder that does nothing

    return waveLetPyr[-1]


    #return pyr[-1]


# Fill this out
# You'll get a numpy array/image of coefficients y
# Return corresponding coefficients x (same shape/size)
# that minimizes (x - y)^2 + lmbda * abs(x)
def denoise_coeff(y,lmbda):
    x1 = y - lmbda/2
    x2 = y + lmbda/2
    x3 = 0
    a1 = customFunc(x1,y,lmbda)  # result matrix of function (x-y)^2 - lmbda|x|
    a2 = customFunc(x2,y,lmbda)
    a3 = customFunc(x3,y,lmbda)
    x = np.zeros(y.shape)
    x = np.where(np.logical_and(a1 < a2, a1 < a3), x1, x)
    x = np.where(np.logical_and(a2 < a1, a2 < a3), x2, x)
    x = np.where(np.logical_and(a3 < a1, a3 < a2), x3, x)

    # #it can be divided into four situations
    # #1. x=0, x=y-lmbda/2 >0 (x>0), x=y+lmbda/2 <0 (x<0) three potential mini points
    # if((y-lmbda/2) > 0 and (y+lmbda/2) < 0 ):
    #     x = min(customFunc1(0,y,lmbda), customFunc1((y-lmbda/2),y,lmbda), customFunc2((y+lmbda/2),y,lmbda))
    #     if(x == customFunc1(0,y,lmbda)):
    #         x=0
    #     if(x == customFunc1((y-lmbda/2),y,lmbda)):
    #         x=y-lmbda/2
    #     if(x == customFunc2((y+lmbda/2),y,lmbda)):
    #         x=y+lmbda/2
    # #2. x=0, x=y-lmbda/2 >0 (x>0) && x=y+lmbda/2 >0 (x<0) two potential mini points
    # if((y-lmbda/2) > 0 and (y+lmbda/2) > 0):
    #     x = min(customFunc1(0,y,lmbda), customFunc1((y-lmbda/2),y,lmbda))
    #     if (x == customFunc1(0, y, lmbda)):
    #         x = 0
    #     if (x == customFunc1((y - lmbda / 2), y, lmbda)):
    #         x = y - lmbda / 2
    # #3. x=0, x=y-lmbda/2 <0 (x>0) && x=y+lmbda/2 <0 (x<0) two potential mini points
    # if ((y - lmbda / 2) < 0 and (y + lmbda / 2) < 0):
    #     x = min(customFunc1(0, y, lmbda), customFunc1((y + lmbda / 2), y, lmbda))
    #     if (x == customFunc1(0, y, lmbda)):
    #         x = 0
    #     if (x == customFunc1((y + lmbda / 2), y, lmbda)):
    #         x = y + lmbda / 2
    # #4. x=0, x=y-lmbda/2 <0 (x>0) && x=y+lmbda/2 >0 (x<0) only x=0 min points
    # if ((y - lmbda / 2) < 0 and (y + lmbda / 2) > 0):
    #     x = 0
    return x

# (x-y)^2 + lmbda|x|
def customFunc(x,y,lmbda):
    return pow(x-y,2) + lmbda*abs(x)

# def judge(x1,x2,x3,y,lmbda):
#     a1 = customFunc(x1, y, lmbda)
#     a2 = customFunc(x2, y, lmbda)
#     a3 = customFunc(x3, y, lmbda)
#     minX = min(a1,a2,a3)
#     if(min == a1):
#         return 1
#     if(min == a2):
#         return 2
#     if(min == a3):
#         return 3

# # x>0
# def customFunc1(x,y,lmbda):
#     return pow(x-y,2) + lmbda*x

# # x<0
# def customFunc2(x,y,lmbda):
#     return pow(x-y,2) - lmbda*x

########################## Support code below

from os.path import normpath as fn # Fixes window/linux path conventions
import warnings
warnings.filterwarnings('ignore')


# Utility functions to clip intensities b/w 0 and 1
# Otherwise imsave complains
def clip(im):
    return np.maximum(0.,np.minimum(1.,im))



############# Main Program

lmain = 0.88

img = np.float32(imread(fn('inputs/p1.png')))/255.
pyr = im2wv(img,4)

for i in range(len(pyr)-1):
    for j in range(2):
        pyr[i][j] = denoise_coeff(pyr[i][j],lmain/(2**i))
    pyr[i][2] = denoise_coeff(pyr[i][2],np.sqrt(2)*lmain/(2**i))
    
im = wv2im(pyr)        
imsave(fn('outputs/prob1.png'),clip(im))
