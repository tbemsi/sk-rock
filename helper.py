fft2c = lambda x: 1/np.sqrt(x.size)*np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(x)))
ifft2c = +lambda x: np.sqrt(x.size)*np.fft.ifftshift(np.fft.ifft2(np.fft.fftshift(x)))

def masked_FFT_t(x,mask):
    """
    
    Returns the transpose operator of the partial FFT transform
    
    INPUT:
        - x: the image whose FFT we want
        - mask: the mask used for the FFT
        
    OUTPUT:
        - the masked FFT of the image
    
    """
    gg = np.zeros(mask.shape,dtype=complex)
    gg[abs(mask)>0]=x
    return ifft2c(gg)


def masked_FFT(x,mask):
    """
    Returns the FFT transform of x
    """
    Rf = fft2c(x)
    return Rf[abs(mask)>0]


def DivergenceIm(matrix_1, matrix_2):
    """
    Computes the divergence between two arrays matrix_1 and matrix_2
    

    INPUTS:
        - matrix_1: numpy matrix
        - matrix_2: numpy matrix
        
    OUTPUTS:
    
        - Divergence between matrix_1 and matrix_2
        
    """
    horiz_difference = matrix_2[:,1:-1] - matrix_2[:,0:-2]
    temp_1 = np.c_[matrix_2[:,0],horiz_difference,-matrix_2[:,-1]]
    
    vert_difference = matrix_1[1:-1, :] - matrix_1[0:-2,:]
    temp_2 = np.c_[matrix_1[0,:],vert_difference.T,-matrix_1[-1,:]]
    temp_2 = temp_2.T
    
    return temp_1 + temp_2


def GradientIm(image):
    """
    Computes the gradient of an image
    
    INPUT:
        - image: the image whose gradient is to be computed
    
    OUTPUT:
        - gradient_x: gradient in the x direction
        - gradient_y: gradient in the y direction
        
    """
    
    horiz_difference = image[1:, :] - image[0:-1,:]
    gradient_x = (np.c_[horiz_difference.T,np.zeros(horiz_difference.shape[1])]).T
        
    vert_difference = image[:,1:] - image[:,0:-1]
    gradient_y = np.c_[vert_difference,np.zeros(vert_difference.shape[0])]
                       
    return  gradient_x, gradient_y


def chambolle_prox_TV(image, approx_parameter, max_iter, tau = 0.249):
    
    """
    Computes the total variation proximal operator of an image according to the
    algorithm proposed by the paper 'An Algorithm for Total Variation Minimization 
    and Applications' by Antonin Chambolle (2004)
    
    INPUTS:
     - image: image
     - approx_parameter: approximation parameter of the proximal algorithm
     - number_iter: number of iterations, usually between 20 and 25 iterations
     - tau: constant inherent to the algorithm
    
    OUTPUTS:
     - the total-variation proximal operator of the image
     
    """
    # initialize
    px = np.zeros(image.shape)
    py = np.zeros(image.shape)      
    number_iterations = 0
    

    while number_iterations < max_iter: 
        number_iterations += 1
        upx, upy = GradientIm(DivergenceIm(px,py) - image/approx_parameter)
        
        temp = np.sqrt(np.multiply(upx,upx) + np.multiply(upy,upy))  
        px = np.divide(px + tau * upx,1 + tau * temp)
        py = np.divide(py + tau * upy,1 + tau * temp)
        

    return image - approx_parameter * DivergenceIm(px,py)


def TVnorm(x):
    """
    Computes the total variation norm of a matrix
    
    INPUTS:
        - x: image
    OUTPUS:
        - tv_norm: total variation norm of x
        
    """
    tvNorm = 0
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            if (j != x.shape[1]-1) and (i != x.shape[0]-1):
                tvNorm += np.sqrt((x[i+1,j]-x[i,j])**2 +(x[i,j+1]-x[i,j])**2)
            if (j == x.shape[1]-1) and (i != x.shape[0]-1):
                tvNorm += abs(x[i+1,j]-x[i,j])
            if (j != x.shape[1]-1) and (i == x.shape[0]-1):
                tvNorm += abs(x[i,j+1]-x[i,j])
    return tvNorm

def LineMask(number_of_angles, dim):
    """
    Generates line mask
    """
    angles = np.linspace(0, np.pi - np.pi/number_of_angles, number_of_angles)
    M = np.zeros((dim, dim))
    #full mask
    for i in range(number_of_angles):
        if ((angles[i] <=np.pi/4) or (angles[i] > 3*np.pi/4)):
            line = np.round(np.tan(angles[i])*np.arange(-dim/2+1, dim/2))+dim/2
            for j in range(dim-1):
                M[int(line[j]), j+1] = 1
        else:
            line = np.round(1/np.tan(angles[i])*np.arange(-dim/2+1, dim/2))+dim/2
            for j in range(dim-1):
                M[j+1, int(line[j])]=1
    return ifftshift(M)

