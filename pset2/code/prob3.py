### Default modules imported. Import more if you need to.
### DO NOT USE linalg.lstsq from numpy or scipy

import numpy as np
from skimage.io import imread, imsave

## Fill out these functions yourself


# Inputs:
#    imgs: A list of N color images, each of which is HxWx3
#    L:    An Nx3 matrix where each row corresponds to light vector
#          for corresponding image.
#    mask: A 0-1 mask of size HxW, showing where observed data is 'valid'.
#
# Returns nrm:
#    nrm: HxWx3 Unit normal vector at each location.
#
# Be careful about division by zero at mask==0 for normalizing unit vectors.
def pstereo_n(imgs, L, mask):
    N = len(imgs)
    grayList = []
    height = imgs[0].shape[0]
    width = imgs[0].shape[1]

    #Create N grayscale Images
    for i in range(N):
        grayImg = (imgs[i][:,:,0] + imgs[i][:,:,1] + imgs[i][:,:,2])/3
        grayList = grayList + [grayImg]

    normal = np.zeros(imgs[0].shape)  # H*W*3 each pixel has a normal (x,y,z), x,y,z respectively save in 3 channel matrix
    #Calculate n
    for x in range(height):
        for y in range(width):
            I = np.zeros((N,1))
            for i in range(N):
                I[i] = grayList[i][x,y]
            n = np.linalg.solve(L.transpose().dot(L),L.transpose().dot(I))
            normal[x,y,0] = n[0]
            normal[x,y,1] = n[1]
            normal[x,y,2] = n[2]

    #normalize the normal
    n_ = np.sqrt(normal[:, :, 0] * normal[:, :, 0] + normal[:, :, 1] * normal[:, :, 1] + normal[:, :, 2] * normal[:, :, 2])
    normalized = np.zeros(imgs[0].shape)
    normalized[:, :, 0] = normal[:, :, 0] / n_ * mask
    normalized[:, :, 1] = normal[:, :, 1] / n_ * mask
    normalized[:, :, 2] = normal[:, :, 2] / n_ * mask

    return normalized


# Inputs:
#    imgs: A list of N color images, each of which is HxWx3
#    nrm:  HxWx3 Unit normal vector at each location (from pstereo_n)
#    L:    An Nx3 matrix where each row corresponds to light vector
#          for corresponding image.
#    mask: A 0-1 mask of size HxW, showing where observed data is 'valid'.
#
# Returns alb:
#    alb: HxWx3 RGB Color Albedo values
#
# Be careful about division by zero at mask==0.
def pstereo_alb(imgs, nrm, L, mask):
    #print(np.where(nrm!=0))
    N = len(imgs)
    Albedos = np.zeros(imgs[0].shape)
    height = imgs[0].shape[0]
    width = imgs[0].shape[1]
    Ln = np.zeros((height,width,N))  ## N 个 l*n构成的矩阵
    for x in range(height):
        for y in range(width):
            n = np.zeros((3,1))
            n[0, 0] = nrm[x, y, 0]
            n[1, 0] = nrm[x, y, 1]
            n[2, 0] = nrm[x, y, 2]
            for i in range(N):
                Ln[x,y,i] = L[i,:].transpose().dot(n)
    #print("Ln first")
    #print(np.where(Ln[:,:,0]!=0))
    denominator = np.zeros((height,width))
    for i in range(N):
        denominator = denominator + Ln[:, :, i] * Ln[:, :, i]

    denominator[np.where(denominator == 0)] = 1
    #print(np.where(denominator!=0))

    for channel in range(3):
        for i in range(N):
            Albedos[:, :, channel] = Albedos[:, :, channel] + imgs[i][:, :, channel] * Ln[:, :, i]

        Albedos[:, :, channel] = Albedos[:, :, channel] / denominator  #some places of denominator are zero

    Albedos[:, :, 0] = Albedos[:, :, 0] * mask
    Albedos[:, :, 1] = Albedos[:, :, 1] * mask
    Albedos[:, :, 2] = Albedos[:, :, 2] * mask

    # for channel in range(imgs[0].shape[2]):
    #     for x in range(height):
    #         for y in range():
    #             numerator = 0
    #             denominator = 0
    #             n = np.zeros((3, 1))  # n is 3*1
    #             n[0, 0] = nrm[x, y, 0]
    #             n[1, 0] = nrm[x, y, 1]
    #             n[2, 0] = nrm[x, y, 2]
    #             for i in range(N):
    #                 numerator = numerator + imgs[i][x,y,channel]*(L[i,:].dot(n))
    #                 denominator = denominator + pow(L[i, :].dot(n), 2)
    #             Albedos[x,y,channel] = numerator/denominator

    return Albedos
    
########################## Support code below

from os.path import normpath as fn # Fixes window/linux path conventions
import warnings
warnings.filterwarnings('ignore')

### Light directions matrix
L = np.float32( \
                [[  4.82962877e-01,   2.58819044e-01,   8.36516321e-01],
                 [  2.50000030e-01,   2.58819044e-01,   9.33012664e-01],
                 [ -4.22219593e-08,   2.58819044e-01,   9.65925813e-01],
                 [ -2.50000000e-01,   2.58819044e-01,   9.33012664e-01],
                 [ -4.82962966e-01,   2.58819044e-01,   8.36516261e-01],
                 [ -5.00000060e-01,   0.00000000e+00,   8.66025388e-01],
                 [ -2.58819044e-01,   0.00000000e+00,   9.65925813e-01],
                 [ -4.37113883e-08,   0.00000000e+00,   1.00000000e+00],
                 [  2.58819073e-01,   0.00000000e+00,   9.65925813e-01],
                 [  4.99999970e-01,   0.00000000e+00,   8.66025448e-01],
                 [  4.82962877e-01,  -2.58819044e-01,   8.36516321e-01],
                 [  2.50000030e-01,  -2.58819044e-01,   9.33012664e-01],
                 [ -4.22219593e-08,  -2.58819044e-01,   9.65925813e-01],
                 [ -2.50000000e-01,  -2.58819044e-01,   9.33012664e-01],
                 [ -4.82962966e-01,  -2.58819044e-01,   8.36516261e-01]])


# Utility functions to clip intensities b/w 0 and 1
# Otherwise imsave complains
def clip(im):
    return np.maximum(0.,np.minimum(1.,im))


############# Main Program


# Load image data
imgs = []
for i in range(L.shape[0]):
    imgs = imgs + [np.float32(imread(fn('inputs/phstereo/img%02d.png' % i)))/255.]

mask = np.float32(imread(fn('inputs/phstereo/mask.png')) > 0)

nrm = pstereo_n(imgs,L,mask)

nimg = nrm/2.0+0.5
nimg = clip(nimg * mask[:,:,np.newaxis])
imsave(fn('outputs/prob3_nrm.png'),nimg)


alb = pstereo_alb(imgs,nrm,L,mask)

alb = alb / np.max(alb[:])
alb = clip(alb * mask[:,:,np.newaxis])

imsave(fn('outputs/prob3_alb.png'),alb)
