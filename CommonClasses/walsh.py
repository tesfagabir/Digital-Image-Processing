import numpy as np

import matplotlib.pyplot as plt
#%matplotlib inline

#import matplotlib.image as img
#import PIL.Image as Image 
from PIL import Image
import math
import cmath

import time

import csv

from numpy import binary_repr

from fractions import gcd

class Walsh(object):
    """
    This class Walsh implements all the procedures for transforming a given 2D digital image
    into its corresponding frequency-domain image (Walsh Transform)
    """
    
    def __init__():
        pass
    
    @classmethod
    def __computeBeta(self, u, x, n):
        uBin = binary_repr(u, width=n)
        xBin = binary_repr(x, width=n)
        beta = 0
        for i in xrange(n):
            beta += (int(xBin[i])*int(uBin[i]))
        
        return beta
    
    #Compute walsh kernel (there is only a single kernel for forward and inverse transform
    #as it is both orthogonal and symmetric).
    @classmethod
    def computeKernel(self, N):
        """
        Computes/generates the walsh kernel function.

        Parameters
        ----------
        N : int
            Size of the kernel to be generated.

        Returns
        -------
        kernel : ndarray
            The generated kernel as a matrix.
        """
        
        #Initialize the kernel
        kernel = np.zeros([N, N])
        #Compute each value of the kernel...
        n = int(math.log(N, 2))        
        for u in xrange(N):
            for x in xrange(N):
                beta = Walsh.__computeBeta(u, x, n)
                kernel[u, x] = (-1)**beta
        
        #To make the kernel orthonormal, we can divide it by sqrt(N)
        #kernel /= math.sqrt(N)
        
        #Return the resulting kernel
        return kernel

    @classmethod
    def computeForwardWalsh(self, imge):
        """
        Computes/generates the 2D Walsh transform.

        Parameters
        ----------
        imge : ndarray
            The input image to be transformed.

        Returns
        -------
        final2DWalsh : ndarray
            The transformed image.
        """
        
        N = imge.shape[0]
        kernel = Walsh.computeKernel(N)

        imge1DWalsh = np.dot(kernel, imge)        
        final2DWalsh = np.dot(imge1DWalsh, kernel)

        return final2DWalsh/N
    
    @classmethod
    def computeInverseWalsh(self, imgeWalsh):
        """
        Computes/generates the inverse of 2D Walsh transform.

        Parameters
        ----------
        imgeWalsh : ndarray
            The Walsh transformed image.

        Returns
        -------
        imgeInverse : ndarray
            The inverse of the transformed image.
        """
        
        N = imgeWalsh.shape[0]
        kernel = Walsh.computeKernel(N)

        imge1DInverse = np.dot(kernel, imgeWalsh)        
        imgeInverse = np.dot(imge1DInverse, kernel)

        return imgeInverse/N
        