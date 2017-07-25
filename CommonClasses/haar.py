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

class Haar(object):
    """
    This class Haar implements all the procedures for transforming a given 2D digital image
    into its corresponding frequency-domain image (Haar Transform)
    """
    
    def __init__():
        pass
        
    #Compute the Haar kernel.
    @classmethod
    def computeKernel(self, N):
        """
        Computes/generates the haar kernel function.

        Parameters
        ----------
        N : int
            Size of the kernel to be generated.

        Returns
        -------
        kernel : ndarray
            The generated kernel as a matrix.
        """
        
        i = 0
        kernel = np.zeros([N, N])
        n = int(math.log(N, 2))

        #Fill for the first row of the kernel
        for j in xrange(N):
            kernel[i, j] = 1.0/math.sqrt(N)


        # For the other rows of the kernel....
        i += 1
        for r in xrange(n):
             for m in xrange(1, (2**r)+1):
                j=0
                for x in np.arange(0, 1, 1.0/N):
                    if (x >= (m-1.0)/(2**r)) and (x < (m-0.5)/(2**r)):
                        kernel[i, j] = (2.0**(r/2.0))/math.sqrt(N)
                    elif (x >= (m-0.5)/(2**r)) and (x < m/(2.0**r)):
                        kernel[i, j] = -(2.0**(r/2.0))/math.sqrt(N)
                    else:
                        kernel[i, j] = 0
                    j += 1
                i += 1
        return kernel

    @classmethod
    def computeForwardHaar(self, imge):
        """
        Computes/generates the 2D Haar transform.

        Parameters
        ----------
        imge : ndarray
            The input image to be transformed.

        Returns
        -------
        final2DHaar : ndarray
            The transformed image.
        """
        
        N = imge.shape[0]
        kernel = Haar.computeKernel(N)

        imge1DHaar = np.dot(kernel, imge) 
        
        #Transpose the kernel as it is not symmetric
        final2DHaar = np.dot(imge1DHaar, kernel.T)

        return final2DHaar/N
    
    @classmethod
    def computeInverseHaar(self, imgeHaar):
        """
        Computes/generates the inverse of 2D Haar transform.

        Parameters
        ----------
        imgeHaar : ndarray
            The Haar transformed image.

        Returns
        -------
        imgeInverse : ndarray
            The inverse of the transformed image.
        """
        
        N = imgeHaar.shape[0]
        kernel = Haar.computeKernel(N)

        imge1DInverse = np.dot(kernel.T, imgeHaar)        
        imgeInverse = np.dot(imge1DInverse, kernel)

        return imgeInverse/N
        