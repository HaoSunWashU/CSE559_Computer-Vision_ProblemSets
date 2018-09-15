## Default modules imported. Import more if you need to.

import numpy as np
from skimage.io import imread, imsave

# Fill this out
# X is input 8-bit grayscale image
# Return equalized image with intensities from 0-255
def histeq(X):
    ########    create a new imgage for output
    img_out = np.zeros(np.shape(X))

    ########    get histogram of input image    ########
    histOfImage = np.histogram(X, range(0,257))
    grayDistribution = histOfImage[0]

    #grayHistogram = np.array(grayDistribution,np.float64)
    grayHistogram = grayDistribution / np.size(X)# gray distribution probability

    ########    calculate accumulated gray distribution probability, and save it in a array
    accumulatedPro = np.zeros(np.size(grayHistogram))  ## 0-255 totally 256

    #### use matrix mutilication
    accumulateMatrix = np.zeros((np.size(accumulatedPro),np.size(accumulatedPro)))

    for x in range(0,256):
        accumulateMatrix += np.eye(np.size(accumulatedPro),np.size(accumulatedPro),x)
    accumulateMatrix = accumulateMatrix.T

    accumulatedPro = np.dot(accumulateMatrix, grayHistogram.T)
    accumulatedPro = accumulatedPro.T
    img_out = [accumulatedPro[x] * 255 for x in X]

    ########    test code for historgram data    ########
    print(np.size(grayDistribution))  # 256 (0-255)
    print(grayDistribution)
    print(grayHistogram)
    print('accumulateMatrix and shape')
    print(accumulateMatrix, np.shape(accumulateMatrix))
    print('accumulatedPro')
    print(accumulatedPro)
    print('the shape of input image X: ')
    print(np.shape(X))
    print('the result of np.histogram')
    print(histOfImage)
    print('grayvalue distribution of image')
    print(grayDistribution)
    print('gray histogram')
    print(grayHistogram)

    return img_out
    

########################## Support code below

from os.path import normpath as fn # Fixes window/linux path conventions
import warnings
warnings.filterwarnings('ignore')

img = imread(fn('inputs/p2_inp.png'))  #get image 8-bit grayscale image

out = histeq(img)  # histogram equalization and return new image. Actually out is a two-dimension array

out = np.maximum(0,np.minimum(255,out))  # set all the num in out into 0-255
out = np.uint8(out)                        # 8-bit image output
imsave(fn('outputs/prob2.png'),out)    # save a new image




##img_tinted = img * [1, 0.95, 0.9]  if a image is colorful, it has RGB channels, so its numpy array shape is (h,w,3)
# 1, 0.95, 0.9will be broadcasting to different channels

