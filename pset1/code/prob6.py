## Default modules imported. Import more if you need to.

import numpy as np
from skimage.io import imread, imsave


## Fill out these functions yourself

def im2wv(img,nLev):
    # Placeholder that does nothing
    height = img.shape[0]
    width = img.shape[1]
    # print(width)
    # for x in range(0, width, 2):
    #     print(x)

    # create list
    list = []  #save different level's images
    list.append(img)
    for i in range(nLev):
        X = list[-1] #get the last one of List
        del list[-1] #remove the last one
        Xnew = np.zeros((int(height/pow(2,i+1)),int(width/pow(2,i+1))))
        H1 = np.zeros((int(height/pow(2,i+1)),int(width/pow(2,i+1))))
        H2 = np.zeros((int(height/pow(2,i+1)),int(width/pow(2,i+1))))
        H3 = np.zeros((int(height/pow(2,i+1)),int(width/pow(2,i+1))))
        for x in range(0, int(width/pow(2,i)),2):
            for y in range(0, int(height/pow(2,i)),2):
                # X[x,y]=a X[x,y+1]=b X[x+1,y]=c X[x+1,y+1]=d
                Xnew[int(x/2),int(y/2)] = (X[x,y] + X[x,y+1] + X[x+1,y] + X[x+1,y+1])/2
                H1[int(x/2),int(y/2)] = (X[x,y+1] + X[x+1,y+1] - X[x,y] - X[x+1,y])/2
                H2[int(x/2),int(y/2)] = (X[x+1,y] + X[x+1,y+1] - X[x,y] - X[x,y+1])/2
                H3[int(x/2),int(y/2)] = (X[x,y] + X[x+1,y+1] - X[x,y+1] - X[x+1,y])/2

        newList = [H1,H2,H3]
        list.append(newList)
        list.append(Xnew)            #put new size X into the last one

    return list

def wv2im(pyr):
    waveLetPyr = pyr
    length = len(waveLetPyr)
    lev = length - 1;
    for i in range(lev):
        L = waveLetPyr[-1]
        del waveLetPyr[-1] #pyr.remove last one
        Hlist = waveLetPyr[-1]
        del waveLetPyr[-1]
        H1 = Hlist[0]
        H2 = Hlist[1]
        H3 = Hlist[2]
        height = L.shape[0]
        width = L.shape[1]
        Xnew = np.zeros((height*2, width*2))
        for x in range(height):
            for y in range(width):
                Xnew[2*x,2*y] = (L[x,y] - H1[x,y] - H2[x,y] + H3[x,y])/2
                Xnew[2*x,2*y+1] = (L[x,y] + H1[x,y] - H2[x,y] - H3[x,y])/2
                Xnew[2*x+1,2*y] = (L[x,y] - H1[x,y] + H2[x,y] - H3[x,y])/2
                Xnew[2*x+1,2*y+1] = (L[x,y] + H1[x,y] + H2[x,y] - H3[x,y])/2
        waveLetPyr.append(Xnew)


    # # Placeholder that does nothing

    return waveLetPyr[-1]



########################## Support code below

from os.path import normpath as fn # Fixes window/linux path conventions
import warnings
warnings.filterwarnings('ignore')


# Utility functions to clip intensities b/w 0 and 1
# Otherwise imsave complains
def clip(im):
    return np.maximum(0.,np.minimum(1.,im))


# Visualize pyramid like in slides
def vis(pyr, lev=0):
    if len(pyr) == 1:
        return pyr[0]/(2**lev)

    sz=pyr[0][0].shape
    sz1 = [sz[0]*2,sz[1]*2]
    img = np.zeros(sz1,dtype=np.float32)

    img[0:sz[0],0:sz[1]] = vis(pyr[1:],lev+1)

    # Just scale / shift gradient images for visualization
    img[sz[0]:,0:sz[1]] = pyr[0][0]*(2**(1-lev))+0.5
    img[0:sz[0],sz[1]:] = pyr[0][1]*(2**(1-lev))+0.5
    img[sz[0]:,sz[1]:] = pyr[0][2]*(2**(1-lev))+0.5

    return img



############# Main Program


img = np.float32(imread(fn('inputs/p6_inp.png')))/255.

# # Visualize pyramids
# pyr = im2wv(img,1)
# imsave(fn('outputs/prob6a_1.png'),clip(vis(pyr)))
#
# pyr = im2wv(img,2)
# imsave(fn('outputs/prob6a_2.png'),clip(vis(pyr)))

pyr = im2wv(img,3)
# imsave(fn('outputs/prob6a_3.png'),clip(vis(pyr)))

pyr1 = pyr
pyr2 = pyr
pyr3 = pyr

print(len(pyr))
print(pyr[0][0].shape)
print(pyr[1][0].shape)
print(pyr[2][0].shape)

# pyr1[0][0][...] = 0.
# pyr1[0][1][...] = 0.
# pyr1[0][2][...] = 0.
# im = clip(wv2im(pyr1))
# imsave(fn('outputs/prob6b_0.png'),im)

# pyr2[0][0][...] = 0.
# pyr2[0][1][...] = 0.
# pyr2[0][2][...] = 0.
#
# pyr2[1][0][...] = 0.
# pyr2[1][1][...] = 0.
# pyr2[1][2][...] = 0.
#
# im = clip(wv2im(pyr2))
# imsave(fn('outputs/prob6b_1.png'),im)

pyr3[0][0][...] = 0.
pyr3[0][1][...] = 0.
pyr3[0][2][...] = 0.

pyr3[1][0][...] = 0.
pyr3[1][1][...] = 0.
pyr3[1][2][...] = 0.

pyr3[2][0][...] = 0.
pyr3[2][1][...] = 0.
pyr3[2][2][...] = 0.

im = clip(wv2im(pyr3))
imsave(fn('outputs/prob6b_2.png'),im)


# # # # # Inverse transform to reconstruct image
# im = clip(wv2im(pyr))
# imsave(fn('outputs/prob6b.png'),im)

# Zero out some levels and reconstruct
# for i in range(len(pyr)-1):
#
#     for j in range(3):
#         pyr[i][j][...] = 0.
#
#     im = clip(wv2im(pyr))
#     imsave(fn('outputs/prob6b_%d.png' % i),im)



