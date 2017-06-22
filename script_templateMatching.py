# -*- coding: utf-8 -*-
"""
Created on Sat Apr  8 01:48:59 2017

@author: cgdmtrn
"""
import cv2
import numpy as np

def ct_calculateHuMoments(img):
    ## this funtion calculates the Hu Moments
    moms = cv2.moments(img)
    huMoms = cv2.HuMoments(moms)
    objHuMoms = -np.sign(huMoms)*np.log10(huMoms)
    objHuMoms[np.isnan(objHuMoms)] = 0
    return objHuMoms

def ct_findObject(stats):
    x = []
    y = []
    w = []
    h = []
    countObj = 0
    ## for each object found
    for lb in range(1,len(stats)):
        ## check whether it consist of higher than a certain pixel number
        ## this number can be changed
        if stats[lb,4] > 20:
            ## obtain location information
            x.append(stats[lb,cv2.CC_STAT_LEFT])
            y.append(stats[lb,cv2.CC_STAT_TOP])
            w.append(stats[lb,cv2.CC_STAT_WIDTH])
            h.append(stats[lb,cv2.CC_STAT_HEIGHT])
            countObj += 1 ## increase the number of objects found
    return x,y,w,h

def ct_isMatching(imgO1,imgO2):
    ## calculate the Hu Moments of two images
    hM1 = ct_calculateHuMoments(imgO1)
    hM2 = ct_calculateHuMoments(imgO2)
    ## find the absolute difference between 7 Hu Moments
    diff = np.abs(hM1 - hM2)
    ## calculate a sum
    sumDiff = diff.sum()
    
    if sumDiff < 1: ## if the difference is less than 1
        return 1 ## it is a match
    else: ## if not
        return 0 ## it is not a match

## load the query image
img = cv2.imread('img.jpg')
imgFinal = img.copy()
imgBinary = cv2.bitwise_not(img[:,:,1])

## find the objects that are connected
output = cv2.connectedComponentsWithStats(imgBinary, 4, cv2.CV_32S)
stats = output[2]
x,y,w,h = ct_findObject(stats)

## prepare logo
logo = cv2.imread('logo.png')
ret,logo = cv2.threshold(logo[:,:,1],127,255,cv2.THRESH_BINARY_INV)

## for each object found with connectedComponents
for obj in range(len(x)):
    imgIns = imgBinary[y[obj]: y[obj] + h[obj], x[obj]: x[obj] + w[obj]]
    ## check whether it is a match
    flag = ct_isMatching(imgIns,logo)
    if flag == 1: ## if it is a match
        ## draw a bounding box
        cv2.rectangle(imgFinal,(x[obj],y[obj]),(x[obj]+w[obj],y[obj]+h[obj]),(0,255,0),2)

## show the image
cv2.imshow('image',imgFinal)
cv2.waitKey(0)
cv2.destroyAllWindows()
