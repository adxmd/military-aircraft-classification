#Adam David
#101235041

import numpy as np
import os
from PIL import Image

import sklearn
from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow import keras

from collections import defaultdict

import os
import shutil

from models import *
from dataGenerator import *

def reformatDataset():
    """
        go through /crop/

        track currentImageNum = 0
        track currentClassID = 1

        go through each folder
            for each image, rename it to image_{currentImageNum} and 
            move to a different directory, 
            increment currentImageNum
        _after_ each folder increase classID
    """


    DATASET_PATH = "/Users/adamdavid/Desktop/Carleton University/4th Year/Winter '25/COMP 4107 A - Neural Networks/Project/aircraftClassification/downloadedDataset"
    DESTINATION_PATH = "/Users/adamdavid/Desktop/Carleton University/4th Year/Winter '25/COMP 4107 A - Neural Networks/Project/aircraftClassification/images"

    #Dict to hold class id corresponding to each aircraft and the number of images we have of that aircraft
    datasetMetadata = defaultdict()
    
    #Get a list of all the folders within /crop/
    directoryList = os.listdir(DATASET_PATH)
    
    #Sort them by alphabetical order
    directoryList.sort()
    directoryList = directoryList[1:]

    #counters to hold the current image number and class id we are going to be working with
    currentImageNum = 0
    currentClassID = 1

    #Open the file we are going to append labels to
    f = open("labels.txt", "a")

    #For each aircraft directory (by alphabetical order, give them a number from 1-80)
    for i in range(len(directoryList)):
        #Get the name of current aircraft we are dealing with
        currentAircraftName = directoryList[i]

        #Create the path to that folder
        currentWorkingPathSTR = DATASET_PATH + f'/{currentAircraftName}'
        
        #List the items in the directory
        innerDirectoryList = os.listdir(currentWorkingPathSTR)

        #The number of pictures we have of that aircraft
        intermNumAircraft = len(innerDirectoryList)

        #Put this data somewhere
        #  key      values
        # int -> (string, int)
        # classID -> (aircraftName, numAircraftPics)
        datasetMetadata[currentClassID] = (currentAircraftName,intermNumAircraft)

        #For each picture in this directory
        for filename in innerDirectoryList:

            if filename.endswith('.jpg'):
                
                #Rename the file
                oldFilePath = os.path.join(currentWorkingPathSTR, filename)
                newFileName = f'image_{currentImageNum}.jpg'
                renamedFilePath = os.path.join(currentWorkingPathSTR, newFileName)
                os.rename(oldFilePath, renamedFilePath)

                #Add a line representing this image's label into the labels.txt file
                f.write(f'{currentClassID}\n')

                #Move the file to a new directory
                destPath = os.path.join(DESTINATION_PATH, newFileName)
                shutil.move(renamedFilePath, destPath)

                print(f'successfully moved {newFileName}')

                currentImageNum += 1

        currentClassID += 1

    #Close the label file
    f.close()
    print(datasetMetadata)

DATASET_METADATA = defaultdict(None, {1: ('A10', 699), 2: ('A400M', 470), 3: ('AG600', 263), 4: ('AH64', 421), 5: ('AV8B', 440), 6: ('An124', 199), 7: ('An22', 94), 8: ('An225', 94), 9: ('An72', 176), 10: ('B1', 630), 11: ('B2', 519), 12: ('B21', 46), 13: ('B52', 581), 14: ('Be200', 296), 15: ('C130', 1422), 16: ('C17', 680), 17: ('C2', 813), 18: ('C390', 151), 19: ('C5', 389), 20: ('CH47', 284), 21: ('CL415', 337), 22: ('E2', 452), 23: ('E7', 210), 24: ('EF2000', 782), 25: ('EMB314', 109), 26: ('F117', 366), 27: ('F14', 522), 28: ('F15', 1459), 29: ('F16', 1776), 30: ('F18', 1595), 31: ('F22', 670), 32: ('F35', 1371), 33: ('F4', 679), 34: ('H6', 412), 35: ('J10', 680), 36: ('J20', 757), 37: ('J35', 34), 38: ('JAS39', 574), 39: ('JF17', 214), 40: ('JH7', 275), 41: ('KAAN', 49), 42: ('KC135', 384), 43: ('KF21', 114), 44: ('KJ600', 52), 45: ('Ka27', 114), 46: ('Ka52', 201), 47: ('MQ9', 359), 48: ('Mi24', 280), 49: ('Mi26', 80), 50: ('Mi28', 147), 51: ('Mi8', 220), 52: ('Mig29', 292), 53: ('Mig31', 467), 54: ('Mirage2000', 442), 55: ('P3', 476), 56: ('RQ4', 297), 57: ('Rafale', 689), 58: ('SR71', 266), 59: ('Su24', 440), 60: ('Su25', 425), 61: ('Su34', 464), 62: ('Su57', 396), 63: ('TB001', 84), 64: ('TB2', 414), 65: ('Tornado', 462), 66: ('Tu160', 377), 67: ('Tu22M', 342), 68: ('Tu95', 367), 69: ('U2', 309), 70: ('UH60', 222), 71: ('US2', 640), 72: ('V22', 849), 73: ('V280', 29), 74: ('Vulcan', 393), 75: ('WZ7', 112), 76: ('XB70', 172), 77: ('Y20', 252), 78: ('YF23', 137), 79: ('Z10', 43), 80: ('Z19', 73)})
NUM_IMAGES = 33872

IMAGE_WIDTH = 224
IMAGE_HEIGHT = 224

#Function used to find the biggest and smallest picture sizes in the dataset
def findMaxMin():
    path = "images"
    maxHeight = 0
    maxWidth = 0

    minHeight = 100000
    minWidth = 100000

    biggestPic = 0
    smallestPic = 10000000000000
    for i in range(33782):
        filename = f'image_{i}.jpg'

        updatedPath = os.path.join(path, filename)

        image = Image.open(updatedPath)
        imageArray = np.asarray(image)

        h,w,c = imageArray.shape

        totalSize = h * w

        if totalSize > biggestPic:
            biggestPic = totalSize
        if totalSize < smallestPic:
            smallestPic = totalSize


        if (h > maxHeight and w > maxWidth):
            maxHeight = h
            maxWidth = w
        if (h < minHeight and w < minWidth):
            minHeight = h
            minWidth = w 

    print(biggestPic, maxHeight, maxWidth, smallestPic, minHeight, minWidth)
    return biggestPic, maxHeight, maxWidth, smallestPic, minHeight, minWidth

# biggest, maxH, maxW, smallest, minH, minW = findMaxMin()

#TESTING MODEL

def testModel():
    trainval_ids, test_ids = sklearn.model_selection.train_test_split(np.arange(NUM_IMAGES), test_size=0.2)

    imageDirectory = "images"

    labelsFilepath = "labels.txt"

    batchSize = 16

    trainDataGen = AircraftDataGenerator(imageDirectory, labelsFilepath, trainval_ids, True, batchSize, IMAGE_HEIGHT, IMAGE_WIDTH, 
                                         effNet=True,
                                         resNet = False, 
                                         inception=False, 
                                         inceptionResNet=False,
                                         mobileNet=False)

    model = efficientNetB3_80(True)
    # model = resnet50_80(trainable=False)
    # model = mobilenetV2_80(trainable=True)
    # model = inceptionv3(trainable=False)
    # model = inception_resnet_v2(trainable = False)

    history = model.fit(x=trainDataGen, epochs=5)

    print(history.history)

    testDataGen = AircraftDataGenerator(imageDirectory, labelsFilepath, test_ids, True, batchSize, IMAGE_HEIGHT, IMAGE_WIDTH, 
                                        effNet=True, 
                                        resNet=False,
                                        inception=False, 
                                        inceptionResNet=False,
                                        mobileNet=False)

    results = model.evaluate(x=testDataGen)

    print(results)

    model.save('efficientnetb3_80_trainable_batch16.keras')


# testModel()

def effNetTo14():


    trainval_ids, test_ids = sklearn.model_selection.train_test_split(np.arange(NUM_IMAGES), test_size=0.2)

    imageDirectory = "images"

    labelsFilepath = "labels.txt"

    batchSize = 16

    trainDataGen = AircraftDataGenerator(imageDirectory, labelsFilepath, trainval_ids, True, batchSize, IMAGE_HEIGHT, IMAGE_WIDTH, 
                                            effNet=True,
                                            resNet = False, 
                                            inception=False, 
                                            inceptionResNet=False,
                                            mobileNet=False)

    testDataGen = AircraftDataGenerator(imageDirectory, labelsFilepath, test_ids, True, batchSize, IMAGE_HEIGHT, IMAGE_WIDTH, 
                                        effNet=True, 
                                        resNet=False,
                                        inception=False, 
                                        inceptionResNet=False,
                                        mobileNet=False)

    model = tf.keras.models.load_model("savedModels/fullyTrainable_BS_16/efficientnetb3_80_trainable_batch16.keras")


    #Test the model on testing set before training again
    results = model.evaluate(x=testDataGen)

    print(f'Evaluation after 5 epochs\n------------------------\n{results}')

    # test after 5 epochs: [0.30405113101005554, 0.9719266891479492]

    #Train for 4 epochs
    model.fit(x=trainDataGen, epochs=4)

    # training thru epoch 6 - 9 not recorded cuz ima dumbass

    history = model.fit(x=trainDataGen, epochs=5)

    # history of training thru epoch 10 - 14: {'accuracy': [0.9779976606369019, 0.9788836240768433, 0.9835351705551147, 0.9862300753593445, 0.985122561454773], 'loss': [0.19533878564834595, 0.1830464005470276, 0.16014009714126587, 0.14364522695541382, 0.14082293212413788]}

    print(f'Training History: \n{history.history}')

    results1 = model.evaluate(x=testDataGen)

    print(f'Evaluation after 7 epochs\n------------------------\n{results1}')

    # test after training for 9 epochs, current total: trained on 14 epochs

    #test results : [0.26504892110824585, 0.9565602540969849]

    print(results)

    model.save('efficientnetb3_80_trainable_batch16_8ep.keras')

