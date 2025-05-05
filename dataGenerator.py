import numpy as np
from PIL import Image

import sklearn
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras

from collections import defaultdict

import os

DATASET_METADATA = defaultdict(None, {1: ('A10', 699), 2: ('A400M', 470), 3: ('AG600', 263), 4: ('AH64', 421), 5: ('AV8B', 440), 6: ('An124', 199), 7: ('An22', 94), 8: ('An225', 94), 9: ('An72', 176), 10: ('B1', 630), 11: ('B2', 519), 12: ('B21', 46), 13: ('B52', 581), 14: ('Be200', 296), 15: ('C130', 1422), 16: ('C17', 680), 17: ('C2', 813), 18: ('C390', 151), 19: ('C5', 389), 20: ('CH47', 284), 21: ('CL415', 337), 22: ('E2', 452), 23: ('E7', 210), 24: ('EF2000', 782), 25: ('EMB314', 109), 26: ('F117', 366), 27: ('F14', 522), 28: ('F15', 1459), 29: ('F16', 1776), 30: ('F18', 1595), 31: ('F22', 670), 32: ('F35', 1371), 33: ('F4', 679), 34: ('H6', 412), 35: ('J10', 680), 36: ('J20', 757), 37: ('J35', 34), 38: ('JAS39', 574), 39: ('JF17', 214), 40: ('JH7', 275), 41: ('KAAN', 49), 42: ('KC135', 384), 43: ('KF21', 114), 44: ('KJ600', 52), 45: ('Ka27', 114), 46: ('Ka52', 201), 47: ('MQ9', 359), 48: ('Mi24', 280), 49: ('Mi26', 80), 50: ('Mi28', 147), 51: ('Mi8', 220), 52: ('Mig29', 292), 53: ('Mig31', 467), 54: ('Mirage2000', 442), 55: ('P3', 476), 56: ('RQ4', 297), 57: ('Rafale', 689), 58: ('SR71', 266), 59: ('Su24', 440), 60: ('Su25', 425), 61: ('Su34', 464), 62: ('Su57', 396), 63: ('TB001', 84), 64: ('TB2', 414), 65: ('Tornado', 462), 66: ('Tu160', 377), 67: ('Tu22M', 342), 68: ('Tu95', 367), 69: ('U2', 309), 70: ('UH60', 222), 71: ('US2', 640), 72: ('V22', 849), 73: ('V280', 29), 74: ('Vulcan', 393), 75: ('WZ7', 112), 76: ('XB70', 172), 77: ('Y20', 252), 78: ('YF23', 137), 79: ('Z10', 43), 80: ('Z19', 73)})
NUM_IMAGES = 33872

IMAGE_WIDTH = 224
IMAGE_HEIGHT = 224


class AircraftDataGenerator(keras.utils.Sequence):
    
    #Initialization / Constructor
    def __init__(self, imageDirectory, labelsPath, desiredIDs, rgb=True, batchSize=64, imageHeight=224, imageWidth=224, effNet=False, resNet=False, inception=False, inceptionResNet = False, mobileNet = False):
        
        #path to the directory that holds all the 33,872 images
        self.imageDirectory = imageDirectory

        #The desired ids limit us to the training/validation or test set
        self.desiredIDs = desiredIDs

        #The list of labels read in from the labels.txt file
        self.labelList = np.loadtxt(labelsPath).squeeze()

        #Whether or not we want to work with RGB images or Grayscale images
        self.rgb = rgb

        #Boolean to hold whether or not we are using efficient net (certain pre-processing must be done if so)
        self.effNet = effNet
        #Boolean to hold whether or not we are using res net (certain pre-processing must be done if so)
        self.resNet = resNet
        #Boolean to hold whether or not we are using inception net (certain pre-processing must be done if so)
        self.inception = inception
        #Boolean to hold whether or not we are using inception res net (certain pre-processing must be done if so)
        self.inceptionResNet = inceptionResNet
         #Boolean to hold whether or not we are using mobile net v2 (certain pre-processing must be done if so)
        self.mobileNet = mobileNet
        
        
        #The batch size
        self.batchSize = batchSize

        #The image shape
        self.imageShape = (imageWidth, imageHeight)

        #Shuffle the desired ids' order every epoch end
        self.on_epoch_end()

    def __len__(self):
        #The number of batches we go through per epoch
        return int(len(self.desiredIDs) / self.batchSize)
    
    #This is called __len__ times per epoch and returns a batch of data of size batchSize
    def __getitem__(self, index):
        
        #Get the image ids of the current batch
        batchIDs = self.desiredIDs[(index * self.batchSize):((index + 1) * self.batchSize)]

        x = None
        y = None

        #for each id in the batch ids
        for currentID in batchIDs:

            #Open the image
            imageFilename = os.path.join(self.imageDirectory, "image_" + str(currentID) + ".jpg")
            image = Image.open(imageFilename).resize(self.imageShape)

            #If we are not working with grayscale images
            if not self.rgb:
                image = image.convert("L")
            
            #convert the image to an np array
            if self.effNet:
                imageArray = tf.keras.applications.efficientnet.preprocess_input(np.expand_dims(np.asarray(image).astype(np.float32), axis=0))
            elif self.inception:
                imageArray = tf.keras.applications.inception_v3.preprocess_input(np.expand_dims(np.asarray(image).astype(np.float32), axis=0))
            elif self.inceptionResNet:
                imageArray = tf.keras.applications.inception_resnet_v2.preprocess_input(np.expand_dims(np.asarray(image).astype(np.float32), axis=0))
            elif self.resNet:
                imageArray = tf.keras.applications.resnet.preprocess_input(np.expand_dims(np.asarray(image).astype(np.float32), axis=0))
            elif self.mobileNet:
                imageArray = tf.keras.applications.mobilenet_v2.preprocess_input(np.expand_dims(np.asarray(image).astype(np.float32), axis=0))
            else:
                imageArray = np.expand_dims(np.asarray(image), axis=0)
                

            #If x is empty, 
            if x is None:
                x = imageArray
                y = self.labelList[currentID] - 1
            #If not, 
            else:
                x = np.vstack((x, imageArray))
                y = np.vstack((y, self.labelList[currentID] - 1))
        
        y = keras.utils.to_categorical(y, num_classes=80)

        return x, y
    
    
    def on_epoch_end(self):
        #Shuffles indexes after each epoch, taken from flower example in class
        np.random.shuffle(self.desiredIDs)
        return
            
#TESTING DATA GENERATOR

def testDataGenerator():
        
    trainval_ids, test_ids = sklearn.model_selection.train_test_split(np.arange(NUM_IMAGES), test_size=0.2)

    imageDirectory = "images"

    labelsFilepath = "labels.txt"

    batchSize = 16

    aircraftDataGen = AircraftDataGenerator(imageDirectory, labelsFilepath, trainval_ids, True, batchSize)

    x, y = aircraftDataGen.__getitem__(2)

# testDataGenerator()
