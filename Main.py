import os
import numpy as np
import time
import sys

from ChexnetTrainer import ChexnetTrainer

#-------------------------------------------------------------------------------- 

def main ():
    
    # runTest()
    runTrain()
  
#--------------------------------------------------------------------------------   

def runTrain():
    
    MOBILENET = 'MOBILE-NET'
    DENSENET121 = 'DENSE-NET-121'
    DENSENET201 = 'DENSE-NET-201'
    RESNET = 'RESNET-50'
    INCEPTIONV3 = 'INCEPTIONV3'
    CONRAD_DENSE_ASPP = 'CONRAD-DENSE-ASPP'
    XCEPTION = 'XCEPTION'

    timestampTime = time.strftime("%H%M%S")
    timestampDate = time.strftime("%d%m%Y")
    timestampLaunch = timestampDate + '-' + timestampTime
    
    #---- Path to the directory with images
    pathDirData =  './database/'
    
    #---- Paths to the files with training, validation and testing sets.
    #---- Each file should contains pairs [path to image, output vector]
    #---- Example: images_011/00027736_001.png 0 0 0 0 0 0 0 0 0 0 0 0 0 0
    pathFileTrain = './dataset/train_1.txt'
    pathFileVal = './dataset/val_1.txt'
    pathFileTest = './dataset/test_1.txt'
    
    #---- Neural network parameters: type of the network, is it pre-trained 
    #---- on imagenet, number of classes
    nnArchitecture = XCEPTION
    nnIsTrained = False
    nnClassCount = 14
    checkpoint = None # './CheXNet14_20_0.73.pth.tar'
    
    #---- Training settings: batch size, maximum number of epochs
    trBatchSize = 16
    trMaxEpoch = 30
    
    #---- Parameters related to image transforms: size of the down-scaled image, cropped image
    imgtransResize = 256#320#256
    imgtransCrop = 224#299 #224
        
    pathModel = 'conrad-dense-aspp-' + timestampLaunch + '.pth.tar'
    
    print ('Training NN architecture = ', nnArchitecture)
    ChexnetTrainer.train(pathDirData, pathFileTrain, pathFileVal, nnArchitecture, nnIsTrained, nnClassCount, trBatchSize, trMaxEpoch, imgtransResize, imgtransCrop, pathModel, checkpoint)
    
    print ('Testing the trained model')
    ChexnetTrainer.test(pathDirData, pathFileTest, pathModel, nnArchitecture, nnClassCount, nnIsTrained, trBatchSize, imgtransResize, imgtransCrop, timestampLaunch)

#-------------------------------------------------------------------------------- 

def runTest():
    
    pathDirData = '/media/lecun/HD/Covid_databases/XRays/ChestX-ray14/images_full/'
    pathFileTest = './Zoozog/dataset/test_1.txt'
    nnArchitecture = 'DENSE-NET-121'
    nnIsTrained = True
    nnClassCount = 14
    trBatchSize = 16
    imgtransResize = 256
    imgtransCrop = 224
    
    pathModel = './densnet_modified-01022021-120722.pth.tar'

    
    timestampLaunch = ''
    
    ChexnetTrainer.test(pathDirData, pathFileTest, pathModel, nnArchitecture, nnClassCount, nnIsTrained, trBatchSize, imgtransResize, imgtransCrop, timestampLaunch)

#-------------------------------------------------------------------------------- 

if __name__ == '__main__':
    main()





