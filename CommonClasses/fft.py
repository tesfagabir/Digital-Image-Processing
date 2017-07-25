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

from dft import DFT

class FFT(object):
    """
    This class FFT implements all the procedures for transforming a given 2D digital image
    into its corresponding frequency-domain image (FFT)
    """
    
    def __init__():
        pass
    
    @classmethod
    def __computeSingleW(self, num, denom):
        """Computes one value of W from the given numerator and denominator values. """
        return math.cos((2*math.pi*num)/denom) - (1j*math.sin((2*math.pi*num)/denom))
    
    @classmethod
    def __computeW(self, val, denom, oneD=True):
        """Computes 1D or 2D values of W from the given numerator and denominator values."""
        if oneD:
            result = np.zeros([val, 1], dtype=np.complex)
            for i in xrange(val):
                result[i] = FFT.__computeSingleW(i, denom)
        else:
            result = np.zeros([val, val], dtype=np.complex)
            for i in xrange(val):
                for j in xrange(val):
                    result[i, j] = FFT.__computeSingleW((i+j), denom)
        return result
        
    @classmethod
    def computeFFT(self, imge):
        """Computes the FFT of a given image.
        """

        #Compute size of the given image
        N = imge.shape[0]

        #Compute the FFT for the base case (which uses the normal DFT)
        if N == 2:
            return DFT.computeForward2DDFTNoSeparability(imge)

        #Otherwise compute FFT recursively

        #Divide the original image into even and odd
        imgeEE = np.array([[imge[i,j] for i in xrange(0, N, 2)] for j in xrange(0, N, 2)]).T
        imgeEO = np.array([[imge[i,j] for i in xrange(0, N, 2)] for j in xrange(1, N, 2)]).T
        imgeOE = np.array([[imge[i,j] for i in xrange(1, N, 2)] for j in xrange(0, N, 2)]).T
        imgeOO = np.array([[imge[i,j] for i in xrange(1, N, 2)] for j in xrange(1, N, 2)]).T

        #Compute FFT for each of the above divided images
        FeeUV = FFT.computeFFT(imgeEE)
        FeoUV = FFT.computeFFT(imgeEO)
        FoeUV = FFT.computeFFT(imgeOE)
        FooUV = FFT.computeFFT(imgeOO)

        #Compute also Ws
        Wu = FFT.__computeW(N/2, N)
        Wv = Wu.T #Transpose
        Wuv = FFT.__computeW(N/2, N, oneD=False)

        #Compute F(u,v) for u,v = 0,1,2,...,N/2  
        imgeFuv = 0.25*(FeeUV + (FeoUV * Wv) + (FoeUV * Wu) + (FooUV * Wuv))

        #Compute F(u, v+M) where M = N/2
        imgeFuMv = 0.25*(FeeUV + (FeoUV * Wv) - (FoeUV * Wu) - (FooUV * Wuv))

        #Compute F(u+M, v) where M = N/2
        imgeFuvM = 0.25*(FeeUV - (FeoUV * Wv) + (FoeUV * Wu) - (FooUV * Wuv))

        #Compute F(u+M, v+M) where M = N/2
        imgeFuMvM = 0.25*(FeeUV - (FeoUV * Wv) - (FoeUV * Wu) + (FooUV * Wuv))

        imgeF1 = np.hstack((imgeFuv, imgeFuvM))
        imgeF2 = np.hstack((imgeFuMv, imgeFuMvM))
        imgeFFT = np.vstack((imgeF1, imgeF2))

        return imgeFFT 
    
    @classmethod
    def computeInverseFFT(self, imgeFFT):
        #return np.real(np.conjugate(FFT.computeFFT(np.conjugate(imgeFFT)*(N**2)))*(N**2))
                #Compute size of the given image
        N = imgeFFT.shape[0]
        return np.real(np.conjugate(FFT.computeFFT(np.conjugate(imgeFFT)*(N**2)))).astype(int)