#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 18 14:51:02 2021

@author: rysul
"""
import numpy as np
import cv2 as cv
from timeit import default_timer as timer
import pandas as pd

# create class of an image with the point operation functions
class PointOperations():
    def __init__(self, fileName): # filename be string
        imageFile = fileName.split('/')[1] # assuming in the immediate directory
        self.imageName = imageFile.split('.')[0]
        self.imageType = imageFile.split('.')[1]
        # read image
        img = cv.imread(fileName) # filename should include the directory as well if not in the same directory
        # make image gray scale assuming all input images are RGB
        imgGray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        # get the dimensions of the image
        self.row, self.column = imgGray.shape
        # stacking out the row values of the pixel
        self.imgGray1D = imgGray.flatten()
        # normalizing the pixel values between 0.0 to 1.0 for log transform
        self.imgGray1DNorm = (self.imgGray1D - np.min(self.imgGray1D))/ (np.max(self.imgGray1D)- np.min(self.imgGray1D))
        self.c = 1.0
        self.scalingF = 255
        
        
    def thresholdFormula(self):
        # using loop flatten
        imgThresh = np.zeros((self.row*self.column,),dtype = int)
        for i in range(self.row*self.column):
            if (self.imgGray1D[i] > 128):
                imgThresh[i] = 255
            else:
                imgThresh[i] = 0
        imgThresh = imgThresh.reshape((self.row, self.column))  
        cv.imwrite('outputs/'+self.imageName +'ThesholdFormula.'+self.imageType,imgThresh)
            
            
    def thresholdLUT(self, LUT):
        # using LUT flatten
        imgThresh = np.zeros((self.row*self.column,),dtype = int)
        for i in range(self.row*self.column):
            imgThresh[i] = LUT[self.imgGray1D[i]]   
        imgThresh = imgThresh.reshape((self.row, self.column))    
        cv.imwrite('outputs/'+self.imageName +'ThesholdLUT.'+self.imageType,imgThresh)
        
    
    def negativeFormula(self):
        imgNeg = np.zeros((self.row*self.column,),dtype = int)
        for i in range(self.row*self.column):
            imgNeg[i] = 255 - self.imgGray1D[i]
        imgNeg = imgNeg.reshape((self.row, self.column))
        cv.imwrite('outputs/'+self.imageName +'NegativeFormula.'+self.imageType,imgNeg)

    def negativeLUT(self, LUT):
        # using LUT Flatten with reshape
        imgNeg = np.zeros((self.row*self.column,),dtype = int)
        for i in range(self.row*self.column):
            imgNeg[i] = LUT[self.imgGray1D[i]]   
        imgNeg = imgNeg.reshape((self.row, self.column))   
        cv.imwrite('outputs/'+self.imageName +'NegativeLUT.'+self.imageType,imgNeg)
        
    
    def logFormula(self):
        imgLog = np.zeros((self.row*self.column,),dtype = int)
        for i in range(self.row*self.column):
            imgLog[i] = self.c * self.scalingF * np.log(1+self.imgGray1DNorm[i])
        imgLog = imgLog.reshape((self.row, self.column,)).astype(int)
        cv.imwrite('outputs/'+self.imageName +'LogFormula.'+self.imageType,imgLog)

    def logLUT(self, LUT):
        imgLog = np.zeros((self.row*self.column,),dtype = int)        
        for i in range(self.row*self.column):
            imgLog[i] = LUT[self.imgGray1D[i]]   
        imgLog = imgLog.reshape((self.row, self.column)).astype(int)
        cv.imwrite('outputs/'+self.imageName +'LogLUT.'+self.imageType,imgLog)
 
    
 
# creates the LUTs        
def createLUT(op):
    LUT = np.zeros((256,),dtype = int)
    if(op == 'threshold'):
        for i in range(256):
            if(i>128):
                LUT[i] = 255
            else:
                LUT[i] = 0
            
    elif(op == 'negative'):
        for i in range(256):
            LUT[i] = 255-i
                
    elif(op == 'log'):
        # have to make some assumptions on min max values otherwise we have to create different LUT for different images
        c = 1.0 
        scaleF = 255.0
        for i in range(256):
            norm = (i-0)/(255-0) # assuming 0 is the min and 255 is max pixel value for all images
            LUT[i] = c * scaleF * np.log(1+norm)
    else:
        assert op == "error", "Operation not found" # to make sure we are writing accurate operation name
            
    return LUT
                
                
if __name__ == '__main__':
    
    imageAmounts = [1, 10, 20, 30, 50] # list of combination of images
    # create dictionary to store the results
    result = {'imageAmount': [], 
              'thresholdFormula':[], 
              'thresholdLUT':[], 
              'negativeFormula': [],
              'negativeLUT':[],
              'logFormula':[], 
              'logLUT':[]
              }
    
    for imageAmount in imageAmounts:
        result['imageAmount'].append(imageAmount)
        print('Number of images: {}'.format(imageAmount))
        imageClassList = []
        # assigning the Pointoperation class to all the images in the dataset which are name 1.jpg, 2.jpg and so on..
        for i in range(imageAmount):
            fileDir = 'dataset/'+str(i+1)+'.jpg'
            imageClass = PointOperations(fileDir)
            imageClassList.append(imageClass)
        
        # thresolding formula
        start = timer()
        for i in range(imageAmount):
            imageClassList[i].thresholdFormula()
        end = timer ()
        result['thresholdFormula'].append(round((end - start),3))
        print("time spent for threshold formula: {:.3f} seconds".format(end-start))
        
        # thresholding LUT
        start = timer()
        threshLUT = createLUT('threshold')
        for i in range(imageAmount):
            imageClassList[i].thresholdLUT(threshLUT)
        end = timer ()
        result['thresholdLUT'].append(round((end - start),3))
        print("time spent for threshold LUT: {:.3f} seconds".format(end-start))
        
        # negative formula
        start = timer()
        for i in range(imageAmount):
            imageClassList[i].negativeFormula()
        end = timer ()
        result['negativeFormula'].append(round((end - start),3))
        print("time spent for negative formula: {:.3f} seconds".format(end-start))
        
        # negative LUT
        start = timer()
        negLUT = createLUT('negative')
        for i in range(imageAmount):
            imageClassList[i].negativeLUT(negLUT)
        end = timer ()
        result['negativeLUT'].append(round((end - start),3))
        print("time spent for negative LUT: {:.3f} seconds".format(end-start))
        
        
        # log formula
        start = timer()
        for i in range(imageAmount):
            imageClassList[i].logFormula()
        end = timer ()
        result['logFormula'].append(round((end - start),3))
        print("time spent for log formula: {:.3f} seconds".format(end-start))
        
        
        # log LUT
        start = timer()
        lLUT = createLUT('log')
        for i in range(imageAmount):
            imageClassList[i].logLUT(lLUT)
        end = timer ()
        result['logLUT'].append(round((end - start),3))
        print("time spent for log LUT: {:.3f}\n".format(end-start))
    
    # convert the result dictionary into a dataframe
    resultDataFrame = pd.DataFrame(result)
    # saving the dataframe into a csv 
    resultDataFrame.to_csv('results/project1Results.csv', index = False)
    
    
    