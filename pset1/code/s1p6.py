def im2wv(img,nLev):

    if nLev == 0:
        return [img]
    
    hA = (img[0::2,:] + img[1::2,:])/2.
    hB = (-img[0::2,:] + img[1::2,:])/2.

    L = hA[:,0::2]+hA[:,1::2]
    h1 = hB[:,0::2]+hB[:,1::2]
    h2 = -hA[:,0::2]+hA[:,1::2]
    h3 = -hB[:,0::2]+hB[:,1::2]


    return [[h1,h2,h3]] + im2wv(L,nLev-1)


def wv2im(pyr):

    while len(pyr) > 1:
        L0 = pyr[-1]

        Hs = pyr[-2]
        H1 = Hs[0]
        H2 = Hs[1]
        H3 = Hs[2]
        
        
        sz = L0.shape
        L = np.zeros([sz[0]*2,sz[1]*2],dtype=np.float32)

        L[::2,::2] = (L0-H1-H2+H3)/2.
        L[1::2,::2] = (L0+H1-H2-H3)/2.
        L[::2,1::2] = (L0-H1+H2-H3)/2.
        L[1::2,1::2] = (L0+H1+H2+H3)/2.
        
        pyr = pyr[:-2] + [L]

    return pyr[0]
