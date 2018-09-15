## Default modules imported. Import more if you need to.
import numpy as np
from scipy.signal import convolve2d as conv2

# Use these as the x and y derivative filters
fx = np.float32([[1,0,-1]]) * np.float32([[1,1,1]]).T / 6.
fy = fx.T


# Compute optical flow using the lucas kanade method
# Use the fx, fy, defined above as the derivative filters
# and compute derivatives on the average of the two frames.
# Also, consider (x',y') values in a WxW window.
# Return two image shape arrays u,v corresponding to the
# horizontal and vertical flow.
def lucaskanade(f1,f2,W):
    u = np.zeros(f1.shape)
    v = np.zeros(f1.shape)
    window = np.ones((W,W))

    f_aver = (f1 + f2) / 2
    Ix = conv2(f_aver, fx, 'same', 'symm')
    Iy = conv2(f_aver, fy, 'same', 'symm')
    It = f2 - f1

    denominator = conv2(Ix ** 2, window, 'same') * conv2(Iy ** 2, window, 'same') - \
                  conv2((Ix * Iy), window, 'same') ** 2
    # denominator[denominator == 0] = np.inf
    u = (-conv2(Iy**2, window, 'same') * conv2((Ix * It), window, 'same') +
         conv2((Ix * Iy), window, 'same') * conv2((Iy * It), window, 'same')) / denominator

    v = (conv2((Ix * It), window, 'same') * conv2((Ix * Iy), window, 'same') -
         conv2(Ix**2, window, 'same') * conv2((Iy * It), window, 'same')) / denominator

    return (u,v)
    
########################## Support code below

from skimage.io import imread, imsave
from os.path import normpath as fn # Fixes window/linux path conventions
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')


f1 = np.float32(imread(fn('inputs/frame10.jpg')))/255.
f2 = np.float32(imread(fn('inputs/frame11.jpg')))/255.

u,v = lucaskanade(f1,f2,11)


# Display quiver plot by downsampling
x = np.arange(u.shape[1])
y = np.arange(u.shape[0])
x,y = np.meshgrid(x,y[::-1])
plt.quiver(x[::8,::8],y[::8,::8],u[::8,::8],-v[::8,::8],pivot='mid')

plt.show()
