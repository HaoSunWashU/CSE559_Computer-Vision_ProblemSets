## Default modules imported. Import more if you need to.

import numpy as np
from skimage.io import imread, imsave


## Fill out these functions yourself

## Take color image, and return 'white balanced' color image
## based on gray world, as described in Problem 2(a). For each
## channel, find the average intensity across all pixels.
##
## Now multiply each channel by multipliers that are inversely
## proportional to these averages, but add upto 3.
def balance2a(img):
    imgR = img[:,:,0]
    imgG = img[:,:,1]
    imgB = img[:,:,2]

    averR = np.sum(imgR)/imgR.size #average
    averG = np.sum(imgG)/imgG.size
    averB = np.sum(imgB)/imgB.size

    alphaR = 1/averR
    alphaG = 1/averG
    alphaB = 1/averB
    print(alphaR,alphaG,alphaB)

    sum = alphaR + alphaG + alphaB
    alphaR = alphaR * 3 / sum
    alphaG = alphaG * 3 / sum
    alphaB = alphaB * 3 / sum

    img[:,:,0] *= alphaR
    img[:,:,1] *= alphaG
    img[:,:,2] *= alphaB


    return img


## Take color image, and return 'white balanced' color image
## based on description in Problem 2(b). In each channel, find
## top 10% of the brightest intensities, take their average.
##
## Now multiply each channel by multipliers that are inversely
## proportional to these averages, but add upto 3.
def balance2b(img):
    imgR = np.sort(img[:, :, 0].reshape(img[:, :, 0].size))
    imgG = np.sort(img[:, :, 1].reshape(img[:, :, 1].size))
    imgB = np.sort(img[:, :, 2].reshape(img[:, :, 2].size))
    print(imgR.size)
    # averR = np.sum(imgR[0:int(0.1*imgR.size)])/int(0.1*imgR.size)
    # averG = np.sum(imgG[0:int(0.1*imgG.size)])/int(0.1*imgG.size)
    # averB = np.sum(imgB[0:int(0.1*imgB.size)])/int(0.1*imgB.size)
    averR = np.sum(imgR[int(0.9 * imgR.size):]) / int(0.1 * imgR.size)
    averG = np.sum(imgG[int(0.9 * imgG.size):]) / int(0.1 * imgG.size)
    averB = np.sum(imgB[int(0.9 * imgB.size):]) / int(0.1 * imgB.size)
    print(averR,averG,averB)
    print(np.sum(imgR[int(0.9*imgR.size):imgR.size]))
    alphaR = 1 / averR
    alphaG = 1 / averG
    alphaB = 1 / averB

    sum = alphaR + alphaG + alphaB
    alphaR = alphaR * 3 / sum
    alphaG = alphaG * 3 / sum
    alphaB = alphaB * 3 / sum
    print(alphaR + alphaG + alphaB)

    img[:, :, 0] *= alphaR
    img[:, :, 1] *= alphaG
    img[:, :, 2] *= alphaB

    return img



########################## Support code below

from os.path import normpath as fn # Fixes window/linux path conventions
import warnings
warnings.filterwarnings('ignore')


# Utility functions to clip intensities b/w 0 and 1
# Otherwise imsave complains
def clip(im):
    return np.maximum(0.,np.minimum(1.,im))



############# Main Program
im1 = np.float32(imread(fn('inputs/CC/ex1.jpg')))/255.
im2 = np.float32(imread(fn('inputs/CC/ex2.jpg')))/255.
im3 = np.float32(imread(fn('inputs/CC/ex3.jpg')))/255.


im1a = balance2a(im1)
im2a = balance2a(im2)
im3a = balance2a(im3)

imsave(fn('outputs/prob2a_1.png'),clip(im1a))
imsave(fn('outputs/prob2a_2.png'),clip(im2a))
imsave(fn('outputs/prob2a_3.png'),clip(im3a))

im1b = balance2b(im1)
im2b = balance2b(im2)
im3b = balance2b(im3)

imsave(fn('outputs/prob2b_1.png'),clip(im1b))
imsave(fn('outputs/prob2b_2.png'),clip(im2b))
imsave(fn('outputs/prob2b_3.png'),clip(im3b))
