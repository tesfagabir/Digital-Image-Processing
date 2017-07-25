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

class Images(object):
    """
    The class Images implements the methods that are related with images.
    These will be used in all the transformation methods.
    """
    
    def __init__():
        pass
    
    @classmethod
    def generateBlackAndWhiteSquareImage(self, imgSize):
        """
        Generates a square-sized black and white image with a given input size.

        Parameters
        ----------
        imgSize : int
            Input number that stores the dimension of the square image to be generated.

        Returns
        -------
        imge : ndarray
            The generated black and white square image.
        """

        #Creating a matrix with a given size where all the stored values are only zeros (for initialization)
        imge = np.zeros([imgSize, imgSize], dtype=int)
        
        #Starting and ending indices of the white part of the image.
        ind1 = imgSize/4
        ind2 = ind1 + (imgSize/2)
        
        #Make a part of the image as white (255)
        imge[ind1:ind2, ind1:ind2] = np.ones([imgSize/2, imgSize/2], dtype=int)*255

        #return the resulting image
        return imge
    
    @classmethod
    def generateImagesWithResizedWhite(self, imge):
        """
        Generates images with the same size as the original but with a resized white part of them.
        """
        
        N = imge.shape[0]
        
        imges = []
        i = N/2
        while i >= 4:
            j = (N - i)/2
            
            #Starting and ending indices for the white part.
            indx1 = j
            indx2 = j+i
            
            #Draw the image.
            imgeNew = np.zeros([N, N],dtype=int)
            imgeNew[indx1:indx2, indx1:indx2] = np.ones([i, i], dtype=int)*255
            
            #Add the image to the list.
            imges.append(imgeNew)
            
            i = i/2
        
        return imges
        
    
    @classmethod
    def resizeImage(self, imge, newSize):        
        """
        Reduces the size of the given image.

        Parameters
        ----------
        imge : ndarray
            Input array that stores the image to be resized.

        Returns
        -------
        newSize : int
            The size of the newly generated image.
        """
            
        #Compute the size of the original image (in this case, only # of rows as it is square)
        N = imge.shape[0]
        
        #The ratio of the original image as compared to the new one.
        stepSize = N/newSize
        
        #Creating a new matrix (image) with a black color (values of zero)
        newImge = np.zeros([N/stepSize, N/stepSize])
        
        #Average the adjacent four pixel values to compute the new intensity value for the new image.
        for i in xrange(0, N, stepSize):
            for j in xrange(0, N, stepSize):
                newImge[i/stepSize, j/stepSize] = np.mean(imge[i:i+stepSize, j:j+stepSize])
        
        #Return the new image
        return newImge
    
    @classmethod
    def generateImages(self, imgSizes=[128, 64, 32, 16, 8]):
        """Generates black and white images with different sizes.
        """
        #Create an empty list of images to save the generated images with different sizes.
        images = []

        #Generate the first and biggest image
        imge = Images.generateBlackAndWhiteSquareImage(imgSizes[0])

        #Add to the images list
        images.append(imge)

        #Generate the resized and smaller images with different sizes.
        for i in range(1, len(imgSizes)):
            size = imgSizes[i]
            images.append(Images.resizeImage(imge, size))

        return images