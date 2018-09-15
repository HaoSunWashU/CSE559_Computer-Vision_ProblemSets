## Default modules imported. Import more if you need to.

import numpy as np
from skimage.io import imread, imsave


## Fill out these functions yourself


# Inputs:
#    nrm: HxWx3. Unit normal vectors at each location. All zeros at mask == 0
#    mask: A 0-1 mask of size HxW, showing where observed data is 'valid'.
#    lmda: Scalar value of lambda to be used for regularizer weight as in slides.
#
# Returns depth map Z of size HxW.
#
# Be careful about division by 0.
#
# Implement in Fourier Domain / Frankot-Chellappa

def ntod(nrm, mask, lmda):
    H = mask.shape[0]
    W = mask.shape[1]

    #get index of mask that is not 0
    index = np.where(mask!=0)
    print(index)
    #get nx ny nz from nrm. nx ny nz (H*W: 734*804)
    nx = nrm[:, :, 0]
    ny = nrm[:, :, 1]
    nz = nrm[:, :, 2]

    #create gx and gy with H*W
    gx = np.zeros(mask.shape)
    gy = np.zeros(mask.shape)
    print(gx.shape)
    gx[index] = -nx[index]/nz[index]
    gy[index] = -ny[index]/nz[index]
    # print(np.where(gx!=0))

    #create fr fx fy
    fr = np.array([[-1/9, -1/9, -1/9],
          [-1/9, 8/9, -1/9],
          [-1/9, -1/9, -1/9]])

    fx = np.array([[0,   0,    0],
          [0.5, 0, -0.5],
          [0,   0,    0]])
    fy = np.array([[0, -0.5, 0],
          [0,    0, 0],
          [0,  0.5, 0]])
    # fx_ft = np.fft.fft2(fx)
    # fy_ft = np.fft.fft2(fy)
    # fr_ft = np.fft.fft2(fr)
    # FX = np.zeros(mask.shape)
    # FY = np.zeros(mask.shape)
    # FR = np.zeros(mask.shape)
    # FX[:2, :2] = fx_ft[1:, 1:]
    # FX[:2, W - 1] = fx_ft[1:, 0]
    # FX[H - 1, :2] = fx_ft[0, 1:]
    # FX[H - 1, W - 1] = fx_ft[0, 0]
    # FY[:2, :2] = fy_ft[1:, 1:]
    # FY[:2, W - 1] = fy_ft[1:, 0]
    # FY[H - 1, :2] = fy_ft[0, 1:]
    # FY[H - 1, W - 1] = fy_ft[0, 0]
    # FR[:2, :2] = fr_ft[1:, 1:]
    # FR[:2, W - 1] = fr_ft[1:, 0]
    # FR[H - 1, :2] = fr_ft[0, 1:]
    # FR[H - 1, W - 1] = fr_ft[0, 0]

    fx_cir = np.zeros(mask.shape)
    fy_cir = np.zeros(mask.shape)
    fr_cir = np.zeros(mask.shape)
    fx_cir[:2, :2] = fx[1:, 1:]
    fx_cir[:2, W - 1] = fx[1:, 0]
    fx_cir[H - 1, :2] = fx[0, 1:]
    fx_cir[H - 1, W - 1] = fx[0, 0]
    fy_cir[:2, :2] = fy[1:, 1:]
    fy_cir[:2, W - 1] = fy[1:, 0]
    fy_cir[H - 1, :2] = fy[0, 1:]
    fy_cir[H - 1, W - 1] = fy[0, 0]
    fr_cir[:2, :2] = fr[1:, 1:]
    fr_cir[:2, W - 1] = fr[1:, 0]
    fr_cir[H - 1, :2] = fr[0, 1:]
    fr_cir[H - 1, W - 1] = fr[0, 0]

    FX = np.fft.fft2(fx_cir)
    FY = np.fft.fft2(fy_cir)
    FR = np.fft.fft2(fr_cir)
    GX = np.fft.fft2(gx)
    GY = np.fft.fft2(gy)
    denominator = np.real(FX) * np.real(FX) + np.imag(FX) * np.imag(FX) + \
                  np.real(FY) * np.real(FY) + np.imag(FY) * np.imag(FY) + \
                  lmda * (np.real(FR) * np.real(FR) + np.imag(FR) * np.imag(FR)) + 1e-12

    # FZ = np.zeros(mask.shape)
    FZ = (np.conj(FX) * GX + np.conj(FY) * GY)/denominator
    Z = np.real(np.where(mask == 1, np.fft.ifft2(FZ),0))
    return Z

########################## Support code below

from os.path import normpath as fn # Fixes window/linux path conventions
import warnings
warnings.filterwarnings('ignore')

import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D


#### Main function

nrm = imread(fn('inputs/phstereo/true_normals.png'))

# Un-comment  next line to read your output instead
# nrm = imread(fn('outputs/prob3_nrm.png'))


mask = np.float32(imread(fn('inputs/phstereo/mask.png')) > 0)

nrm = np.float32(nrm/255.0)
nrm = nrm*2.0-1.0
nrm = nrm * mask[:,:,np.newaxis]


# Main Call
Z = ntod(nrm,mask,1e-6)


# Plot 3D shape

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

x,y = np.meshgrid(np.float32(range(nrm.shape[1])),np.float32(range(nrm.shape[0])))
x = x - np.mean(x[:])
y = y - np.mean(y[:])

Zmsk = Z.copy()
Zmsk[mask == 0] = np.nan
Zmsk = Zmsk - np.nanmedian(Zmsk[:])

lim = 100
ax.plot_surface(x,-y,Zmsk, \
                linewidth=0,cmap=cm.inferno,shade=True,\
                vmin=-lim,vmax=lim)

ax.set_xlim3d(-450,450)
ax.set_ylim3d(-450,450)
ax.set_zlim3d(-450,450)

plt.show()
