    ## Default modules imported. Import more if you need to.

import numpy as np
from skimage.io import imread, imsave
from scipy.signal import convolve2d as conv2


# Fill this out
def kernpad(K,size):# size is the shape of img
        Ko = np.zeros(size,dtype=np.float32)
        height = K.shape[0]
        width = K.shape[1]
        imgHeight = size[0]
        imgWidth = size[1]

        #print(size)
        #print(Ko.shape)
        #print(K.shape)
        #print(int(width/2 + 1))

        if(width%2==0):
            K1 = np.zeros((height/2,width/2), dtype = np.float32)
            K2 = np.zeros((height/2, width/2), dtype=np.float32)
            K3 = np.zeros((height/2,width/2), dtype = np.float32)
            K4 = np.zeros((height/2, width/2), dtype=np.float32)
            K1 = K[0:height/2,0:width/2]
            K2 = K[height/2:height,0:width/2]
            K3 = K[0:height/2,width/2:width]
            K4 = K[height/2:height,width/2:width]
            K1 = np.rot90(K1,2)
            K2 = np.rot90(K2,2)
            K3 = np.rot90(K3,2)
            K4 = np.rot90(K4,2)
            Ko[0:height/2, 0:width/2] = K1
            Ko[imgHeight-height/2:imgHeight, 0:width/2] = K2
            Ko[0:height/2, imgWidth-width/2:imgWidth] = K3
            Ko[imgHeight-height/2:imgHeight, imgWidth-width/2] = K4

        if(width%2!=0):
            K1 = np.zeros((int(height/2+1),int(width/2+1)),dtype = np.float32)
            K2 = np.zeros((int(height/2+1),int(width/2)),dtype = np.float32)
            K3 = np.zeros((int(height/2),int(width/2+1)),dtype = np.float32)
            K4 = np.zeros((int(height/2),int(width/2)),dtype = np.float32)
            K1 = K[0:int(height/2+1), 0:int(width/2+1)]
            K2 = K[int(height/2+1):height, 0:int(width/2+1)]
            K3 = K[0:int(height/2+1), int(width/2+1):width]
            K4 = K[int(height/2+1):height, int(width/2+1):width]
            K1 = np.rot90(K1, 2)
            K2 = np.rot90(K2, 2)
            K3 = np.rot90(K3, 2)
            K4 = np.rot90(K4, 2)
            Ko[0:int(height / 2 + 1), 0:int(width / 2 + 1)] = K1
            Ko[imgHeight - int(height / 2 + 1) + 1:imgHeight, 0 : int(width / 2 + 1)] = K2
            Ko[0:int(height / 2 + 1), imgWidth - int(width / 2 + 1) + 1:imgWidth] = K3
            Ko[imgHeight - int(height / 2 + 1) + 1:imgHeight, imgWidth - int(width / 2 + 1) + 1:imgWidth] = K4

        #print(K1)
        # print(K1.shape)


        return Ko

    ########################## Support code below

from os.path import normpath as fn # Fixes window/linux path conventions
import warnings
warnings.filterwarnings('ignore')

img = np.float32(imread(fn('inputs/p5_inp.png')))/255.

# Create Gaussian Kernel
x = np.float32(range(-21,22))
x,y = np.meshgrid(x,x)
G = np.exp(-(x*x+y*y)/2/9.)
G = G / np.sum(G[:])

# Traditional convolve
v1 = conv2(img,G,'same','wrap')

# Convolution in Fourier domain
G = kernpad(G,img.shape)
v2f = np.fft.fft2(G)*np.fft.fft2(img)
v2 = np.real(np.fft.ifft2(v2f))

# Stack them together and save
out = np.concatenate([img,v1,v2],axis=1)
out = np.minimum(1.,np.maximum(0.,out))

imsave(fn('outputs/prob5.png'),out)



