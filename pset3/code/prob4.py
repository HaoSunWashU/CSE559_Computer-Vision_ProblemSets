## Default modules imported. Import more if you need to.

import numpy as np
import math

## Fill out these functions yourself

# Fits a homography between pairs of pts
#   pts: Nx4 array of (x,y,x',y') pairs of N >= 4 points
# Return homography that maps from (x,y) to (x',y')
#
# Can use np.linalg.svd
def getH(pts):
    #get two arrays for (x,y) and (x',y') points
    p = np.zeros((pts.shape[0],3))
    p1 = np.zeros((pts.shape[0],3))
    p[:,0:2] = pts[:,0:2]
    p[:,2] = 1
    p = p.transpose()
    print(p)
    p1[:,0:2] = pts[:, 2:]
    p1[:, 2] = 1
    p1 = p1.transpose()
    print(p1)

    # computer A
    A = np.zeros((3*pts.shape[0],9))  # 12 * 9
    for i in range(pts.shape[0]):  #pts.shape[0] = 4
        A1 = np.array([[0,       -p1[2,i],    p1[1,i]],
                      [p1[2,i],   0,          -p1[0,i]],
                      [-p1[1,i],   p1[0,i],    0]])

        A2 = np.zeros((3,9))
        A2[0,0:3] = p[:,i].transpose()
        A2[1,3:6] = p[:,i].transpose()
        A2[2,6:] =  p[:,i].transpose()

        A[ (3*i) : (3*i + 3), :] = A1.dot(A2)

    #get h and reshape to H
    U,D,V = np.linalg.svd(A)
    h = V[-1]
    print(h)
    H = h.reshape((3,3))
    print(H)
    rms = math.sqrt(np.sum(H*H))
    print(rms)
    H = H/rms
    print(H)
#(np.sqrt(np.sum(H*H)/9))
    # standarlize
    return H

    

# Splices the source image into a quadrilateral in the dest image,
# where dpts in a 4x2 image with each row giving the [x,y] co-ordinates
# of the corner points of the quadrilater (in order, top left, top right,
# bottom left, and bottom right).
#
# Note that both src and dest are color images.
#
# Return a spliced color image.
def splice(src,dest,dpts):
    height = src.shape[0]
    width = src.shape[1]
    # print(H, W)
    # spts = np.float32([[0, 0], [height - 1, 0], [0, width - 1], [height - 1, width - 1]])
    spts = np.float32([[0, 0], [0,width - 1], [height - 1,0], [height - 1, width]])
    pts = np.zeros((4, 4))
    pts[:, 0:2] = dpts
    pts[:, 2:] = spts

    # print(pts)
    H = getH(pts)

    H_dest = dest.shape[0]
    W_dest = dest.shape[1]

    for x in range(H_dest):
        for y in range(W_dest):
            p = np.float32([x,y,1])
            p = p.transpose()
            p1 = H.dot(p) # x', y', 1?
            p1 = p1.transpose()
            p1x = p1[0] / p1[2]
            p1y = p1[1] / p1[2]

            # within the scope of src
            # if((p1[0]/p1[2] >= 0 and p1[0] < height-1) and (p1[1] >= 0 and p1[1] < width -1)):
            if ((p1x >= 0 and p1x < height - 1) and (p1y >= 0 and p1y < width - 1)):
                # a = ((p1[0] * 10) % 10)/10
                # b = ((p1[1] * 10) % 10)/10
                a = ((p1x * 10) % 10) / 10  #first weight
                b = ((p1y * 10) % 10) / 10  #second weight
                a1 = a/(a+b)
                b1 = b/(a+b)
                i = int(p1x)  # x
                j = int(p1y)  # y

                # i = int(p1[0])  # x
                # j = int(p1[1])  # y
                intensity = b1*(a1*src[i, j] + b1*src[i, j+1]) + a1*(a1*src[i+1,j] + b1*src[i+1,j+1])
                dest[x,y] = intensity

    return dest
    
    
########################## Support code below

from skimage.io import imread, imsave
from os.path import normpath as fn # Fixes window/linux path conventions
import warnings
warnings.filterwarnings('ignore')

simg = np.float32(imread(fn('inputs/p4src.png')))/255.
dimg = np.float32(imread(fn('inputs/p4dest.png')))/255.
# dpts = np.float32([ [276,54],[406,79],[280,182],[408,196]]) # Hard coded top-left, bottom-left, top-right, bottom-right
dpts = np.float32([[54,276],[79,406],[182,280],[196,408]])

comb = splice(simg,dimg,dpts)

imsave(fn('outputs/prob4.png'),comb)
