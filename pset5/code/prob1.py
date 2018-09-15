## Default modules imported. Import more if you need to.

import numpy as np
from scipy.signal import convolve2d as conv2

def get_cluster_centers(im,num_clusters):
    # Implement a method that returns an initial grid of cluster centers. You should first
    # create a grid of evenly spaced centers (hint: np.meshgrid), and then use the method
    # discussed in class to make sure no centers are initialized on a sharp boundary.
    # You can use the get_gradients method from the support code below.

    cluster_centers = np.zeros((num_clusters,2),dtype='int')

    """ YOUR CODE GOES HERE """
    H = im.shape[0] # 250*250
    W = im.shape[1]
    K = num_clusters
    S = ((H*W)/K)**0.5 # length of grid side
    im_gradients = get_gradients(im)
    # neighbors = np.zeros((3,3))
    # create initial K cluster centers
    y = np.linspace(S/2, H-S/2, H/S)   # create y indexs
    x = np.linspace(S/2, W-S/2, W/S)   # create x indexs
    y_index, x_index = np.meshgrid(y,x)
    y_index = np.asarray(y_index, dtype='int').reshape(K)
    x_index = np.asarray(x_index, dtype='int').reshape(K)
    cluster_centers[:,0] = y_index
    cluster_centers[:,1] = x_index

    # adjust based on 3*3 neighbors' gradients
    for point in range(num_clusters):  #iterate each point index in cluster_centers
        i, j = cluster_centers[point,:]
        i_gradient, j_gradient = np.where(im_gradients[i-1:i+2,j-1:j+2] == np.min(im_gradients[i-1:i+2,j-1:j+2]))
        cluster_centers[point, :] = i + (i_gradient[0] - 1), j + (j_gradient[0] - 1)

    # print(cluster_centers)
    return cluster_centers

def slic(im,num_clusters,cluster_centers):
    # Implement the slic function such that all pixels assigned to a label
    # should be close to each other in squared distance of augmented vectors.
    # You can weight the color and spatial components of the augmented vectors
    # differently. To do this, experiment with different values of spatial_weight.
    h,w,c = im.shape
    clusters = np.zeros((h,w))  #label[n]

    """ YOUR CODE GOES HERE """

    #initialization
    K = num_clusters
    min_dist = np.full((h, w), np.inf)  #min_dist
    S = ((h*w)/K)**0.5                  # length of grid side
    I_ = np.zeros((h, w, 5))            # R5 augmented vector
    I_[:,:,0:3] = im                    # first 3 dimensions of a vector  0, 1, 2

    iter_num = 10                       # times for iteration
    spatialWeight = 0.3                 # spatial weight  \alpha
    x = np.zeros((h,w))
    y = np.zeros((h,w))
    x_index = np.arange(w)
    y_index = np.arange(h)
    x[:, 0:w] = x_index
    y[0:h,:] = y_index
    y = y.T
    I_[:,:,3] = y * spatialWeight       #fourth dimension
    I_[:,:,4] = x * spatialWeight       #fifth dimension

    #initialize uk, initial value equal to augmented vector of initial cluster center
    uks = np.zeros((K,5))
    for cluster in range(K):
        i, j = cluster_centers[cluster, :]
        uks[cluster] = I_[i, j, :]  # cluter center augmented vector

    #iteration
    for iter in range(iter_num):

        # operation for each cluster_center
        for cluster in range(K):
            #one cluster_center
            i, j = cluster_centers[cluster, :]
            uk = uks[cluster]               #cluter center augmented vector
            # determine the range of 2S*2S neighbor area
            y1 = max(0, i - int(S))
            y2 = min(h, i + int(S))
            x1 = max(0, j - int(S))
            x2 = min(w, j + int(S))

            A = I_[y1 : y2, x1 : x2, :] - uk
            B = np.sum(A*A, axis=2)             # cost to uk
            C = B - min_dist[y1 : y2, x1: x2]   # compare with min_dist
            index_relative = np.where(C < 0)             # index that cost less than min_dist
            index_absolute0 = index_relative[0] + y1
            index_absolute1 = index_relative[1] + x1

            # #renew min_dist and clusters
            min_dist[index_absolute0, index_absolute1] = B[index_relative]
            clusters[index_absolute0, index_absolute1] = cluster

        # renew cluster_centers and uks
        for cluster in range(K):
            index_cluster = np.where(clusters == cluster)
            uks[cluster] = np.mean(I_[index_cluster, :])
            i = uks[cluster][3]/spatialWeight
            j = uks[cluster][4]/spatialWeight
            cluster_centers[cluster] = i,j

    return clusters

########################## Support code below

from skimage.io import imread, imsave
from os.path import normpath as fn # Fixes window/linux path conventions
import matplotlib.cm as cm
import warnings
warnings.filterwarnings('ignore')

# Use get_gradients (code from pset1) to get the gradient of your image when initializing your cluster centers.
def get_gradients(im):
    if len(im.shape) > 2:
        im = np.mean(im,axis=2)
    df = np.float32([[1,0,-1]])
    sf = np.float32([[1,2,1]])
    gx = conv2(im,sf.T,'same','symm')
    gx = conv2(gx,df,'same','symm')
    gy = conv2(im,sf,'same','symm')
    gy = conv2(gy,df.T,'same','symm')
    return np.sqrt(gx*gx+gy*gy)

# normalize_im normalizes our output to be between 0 and 1
def normalize_im(im):
    im += np.abs(np.min(im))
    im /= np.max(im)
    return im

# create an output image of our cluster centers
def create_centers_im(im,centers):
    for center in centers:
        im[center[0]-2:center[0]+2,center[1]-2:center[1]+2] = [255.,0.,255.]
    return im

im = np.float32(imread(fn('inputs/lion.jpg')))

num_clusters = [25,49,64,81,100]
for num_clusters in num_clusters:
    cluster_centers = get_cluster_centers(im,num_clusters)
    imsave(fn('outputs/prob1a_' + str(num_clusters)+'_centers.jpg'),normalize_im(create_centers_im(im.copy(),cluster_centers)))
    out_im = slic(im,num_clusters,cluster_centers)

    Lr = np.random.permutation(num_clusters)
    out_im = Lr[np.int32(out_im)]
    dimg = cm.jet(np.minimum(1,np.float32(out_im.flatten())/float(num_clusters)))[:,0:3]
    dimg = dimg.reshape([out_im.shape[0],out_im.shape[1],3])
    imsave(fn('outputs/prob1b_'+str(num_clusters)+'.jpg'),normalize_im(dimg))
