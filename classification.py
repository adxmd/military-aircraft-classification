#Adam David
#101235041

import numpy as np
import os
from PIL import Image

import sklearn
from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow import keras

import matplotlib.pyplot as plt

from collections import defaultdict

import os
import shutil

DATASET_PATH = "/Users/adamdavid/Desktop/Carleton University/4th Year/Winter '25/COMP 4107 A - Neural Networks/Project/aircraftClassification/downloadedDataset"
DESTINATION_PATH = "/Users/adamdavid/Desktop/Carleton University/4th Year/Winter '25/COMP 4107 A - Neural Networks/Project/aircraftClassification/images"

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

    imageDirectory = "/Users/adamdavid/Desktop/Carleton University/4th Year/Winter '25/COMP 4107 A - Neural Networks/Project/aircraftClassification/images"

    labelsFilepath = "/Users/adamdavid/Desktop/Carleton University/4th Year/Winter '25/COMP 4107 A - Neural Networks/Project/aircraftClassification/labels.txt"

    batchSize = 16

    aircraftDataGen = AircraftDataGenerator(imageDirectory, labelsFilepath, trainval_ids, True, batchSize)

    x, y = aircraftDataGen.__getitem__(2)

#Function used to find the biggest and smallest picture sizes in the dataset
def findMaxMin():
    path = "/Users/adamdavid/Desktop/Carleton University/4th Year/Winter '25/COMP 4107 A - Neural Networks/Project/aircraftClassification/images"
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

def customCNN(learningRate = 0.01):
    """
    Convolutional Neural Network

    k: kernel size
    s: stride
    p: padding
    a: activation function

    Architecutre:
      Input:                      size: (224, 224, 3)(RGB)
      Convolution Layer 1:        filters:3->32, k=3, s=1, p=1, a=ReLU, batchNormalization  
      Conovlution Layer 2:        filters:32->64, k=3, s=1, p=1, a=ReLU, batchNormalization, maxPooling   
      Conovlution Layer 3:        filters:64->128, k=3, s=1, p=1, a=ReLU, batchNormalization, maxPooling   
      Conovlution Layer 4:        filters:128->256, k=3, s=1, p=1, a=ReLU, batchNormalization, maxPooling   
      Conovlution Layer 5:        filters:256->512, k=3, s=1, p=1, a=ReLU, batchNormalization, maxPooling   

      Max Pooling Layer:          poolSize=2, s=2, p=0

      Global Average Pooling:     

      Dropout:                    0.5
      Fully Connected 1 Layer:    256 neurons,  a=ReLU
      Output Layer:               80 neurons,   a=softmax
    
    Hyper-Parameters:
      Learning Rate:  0.01 (default)
      Optimizer:      Adaptive Moment Estimation (Adam) Optimizer
      Loss:           Categorical Cross-Entropy Loss

    """
    #Model created using Keras Functional API
    #Layers

    #224 x 224 x 3 - Input
    inputLayer = tf.keras.Input(shape=(IMAGE_HEIGHT,IMAGE_WIDTH,3))

    #Convolution Blocks
    convLayer1 = tf.keras.layers.Conv2D(filters=32, kernel_size=3, strides=1, padding="same", activation="relu", kernel_initializer='he_normal')

    convLayer2 = tf.keras.layers.Conv2D(filters=64, kernel_size=3, strides=1, padding="same", activation="relu", kernel_initializer='he_normal')

    convLayer3 = tf.keras.layers.Conv2D(filters=128, kernel_size=3, strides=1, padding="same", activation="relu", kernel_initializer='he_normal')

    convLayer4 = tf.keras.layers.Conv2D(filters=256, kernel_size=3, strides=1, padding="same", activation="relu", kernel_initializer='he_normal')

    convLayer5 = tf.keras.layers.Conv2D(filters=512, kernel_size=3, strides=1, padding="same", activation="relu", kernel_initializer='he_normal')
    
    #Max Pooling Layer
    poolLayer = tf.keras.layers.MaxPool2D(pool_size=2,strides=2,padding="valid")

    #Batch Normalization
    batchNormLayer1 = tf.keras.layers.BatchNormalization()
    batchNormLayer2 = tf.keras.layers.BatchNormalization()
    batchNormLayer3 = tf.keras.layers.BatchNormalization()
    batchNormLayer4 = tf.keras.layers.BatchNormalization()
    batchNormLayer5 = tf.keras.layers.BatchNormalization()


    #Global Average Pooling Layer for variable-sized inputs
    gapLayer = tf.keras.layers.GlobalAveragePooling2D()

    dropoutLayer = tf.keras.layers.Dropout(0.5)

    #using GAP instead
    #Flatten input to be inputted into fully-connected feedforward network 
    # flattenLayer = tf.keras.layers.Flatten()

    # denseLayer1 = tf.keras.layers.Dense(1024, activation="relu")
    denseLayer1 = tf.keras.layers.Dense(256, activation="relu", kernel_initializer="he_normal")

    #Output layer
    outputLayer = tf.keras.layers.Dense(80, activation="softmax")

    #Forward Pass
    conv1 = convLayer1(inputLayer)
    conv1b = batchNormLayer1(conv1)

    conv2 = convLayer2(conv1b)
    conv2b = batchNormLayer2(conv2)
    conv2p = poolLayer(conv2b)

    conv3 = convLayer3(conv2p)
    conv3b = batchNormLayer3(conv3)
    conv3p = poolLayer(conv3b)

    conv4 = convLayer4(conv3p)
    conv4b = batchNormLayer4(conv4)
    conv4p = poolLayer(conv4b)

    conv5 = convLayer5(conv4p)
    conv5b = batchNormLayer5(conv5)
    conv5p = poolLayer(conv5b)

    gap = gapLayer(conv5p)

    dropout1 = dropoutLayer(gap)
    dense1 = denseLayer1(dropout1)

    output = outputLayer(dense1)

    #Create the model
    model = tf.keras.Model(inputs=inputLayer, outputs=output)

    #Adaptive Moment Estimation Optimizer with Learning Rate (default=0.01)
    optimizer = keras.optimizers.Adam(learning_rate = learningRate)

    # Compile the model with Adam and Categorical Cross-Entropy Loss
    model.compile(optimizer=optimizer, 
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    # Print a summary of the model
    print(model.summary())
    
    return model

def efficientNetB1_80():
    #EfficientNetB1 expects inputs to be dtype=float32 but the data generator currently generates dtype=uint8
    input = tf.keras.Input(shape=(224,224,3))

    # preprocessLayer = tf.keras.layers.Lambda(lambda img: tf.keras.aplications.efficientnet.preprocess_input(img))

    baseModel = tf.keras.applications.EfficientNetB1(include_top=False, weights="imagenet", input_shape=(224,224,3), pooling="max")

    batchNormLayer = tf.keras.layers.BatchNormalization()

    denseLayer = tf.keras.layers.Dense(256, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(0.01))

    dropoutLayer = tf.keras.layers.Dropout(0.2)

    outputLayer = tf.keras.layers.Dense(80, activation="softmax")

    # float32 = preprocessLayer(input)
    base = baseModel(input)
    batchNorm = batchNormLayer(base)
    dense = denseLayer(batchNorm)
    dropout = dropoutLayer(dense)
    output = outputLayer(dropout)

    model = tf.keras.Model(inputs=input, outputs=output)

    model.compile(optimizer=tf.keras.optimizers.Adamax(learning_rate=0.001),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    
    model.summary()

    return model

def efficientNetB2_80():
    input = tf.keras.Input(shape=(224,224,3))

    baseModel = tf.keras.applications.EfficientNetB2(include_top=False, weights="imagenet", input_shape=(224,224,3), pooling="max")

    batchNormLayer = tf.keras.layers.BatchNormalization()

    denseLayer = tf.keras.layers.Dense(256, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(0.01))

    dropoutLayer = tf.keras.layers.Dropout(0.2)

    outputLayer = tf.keras.layers.Dense(80, activation="softmax")

    base = baseModel(input)
    batchNorm = batchNormLayer(base)
    dense = denseLayer(batchNorm)
    dropout = dropoutLayer(dense)
    output = outputLayer(dropout)

    model = tf.keras.Model(inputs=input, outputs=output)

    model.compile(optimizer=tf.keras.optimizers.Adamax(learning_rate=0.001),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    
    model.summary()

    return model

def efficientNetB3_80(trainable = False):

    """
    When frozen:
    ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━┓
    ┃ Layer (type)                    ┃ Output Shape           ┃       Param # ┃
    ┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━┩
    │ input_layer (InputLayer)        │ (None, 224, 224, 3)    │             0 │
    ├─────────────────────────────────┼────────────────────────┼───────────────┤
    │ efficientnetb3 (Functional)     │ (None, 1536)           │    10,783,535 │
    ├─────────────────────────────────┼────────────────────────┼───────────────┤
    │ batch_normalization             │ (None, 1536)           │         6,144 │
    │ (BatchNormalization)            │                        │               │
    ├─────────────────────────────────┼────────────────────────┼───────────────┤
    │ dense (Dense)                   │ (None, 256)            │       393,472 │
    ├─────────────────────────────────┼────────────────────────┼───────────────┤
    │ dropout (Dropout)               │ (None, 256)            │             0 │
    ├─────────────────────────────────┼────────────────────────┼───────────────┤
    │ dense_1 (Dense)                 │ (None, 80)             │        20,560 │
    └─────────────────────────────────┴────────────────────────┴───────────────┘
    Total params: 11,203,711 (42.74 MB)
    Trainable params: 417,104 (1.59 MB)
    Non-trainable params: 10,786,607 (41.15 MB) 

    """

    input = tf.keras.Input(shape=(224,224,3))

    baseModel = tf.keras.applications.EfficientNetB3(include_top=False, weights="imagenet", input_shape=(224,224,3), pooling="max")

    baseModel.trainable = trainable

    batchNormLayer = tf.keras.layers.BatchNormalization()

    denseLayer = tf.keras.layers.Dense(256, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(0.01))

    dropoutLayer = tf.keras.layers.Dropout(0.2)

    outputLayer = tf.keras.layers.Dense(80, activation="softmax")

    base = baseModel(input)
    batchNorm = batchNormLayer(base)
    dense = denseLayer(batchNorm)
    dropout = dropoutLayer(dense)
    output = outputLayer(dropout)

    model = tf.keras.Model(inputs=input, outputs=output)

    model.compile(optimizer=tf.keras.optimizers.Adamax(learning_rate=0.001),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    
    model.summary()

    return model

def resnet50_80(trainable = False):

    """
    ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━┓
    ┃ Layer (type)                         ┃ Output Shape                ┃         Param # ┃
    ┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━┩
    │ input_layer (InputLayer)             │ (None, 224, 224, 3)         │               0 │
    ├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
    │ resnet50 (Functional)                │ (None, 2048)                │      23,587,712 │
    ├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
    │ batch_normalization                  │ (None, 2048)                │           8,192 │
    │ (BatchNormalization)                 │                             │                 │
    ├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
    │ dense (Dense)                        │ (None, 256)                 │         524,544 │
    ├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
    │ dropout (Dropout)                    │ (None, 256)                 │               0 │
    ├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
    │ dense_1 (Dense)                      │ (None, 80)                  │          20,560 │
    └──────────────────────────────────────┴─────────────────────────────┴─────────────────┘
    Total params: 24,141,008 (92.09 MB)
    Trainable params: 549,200 (2.10 MB)
    Non-trainable params: 23,591,808 (90.00 MB)
    
    """

    input = tf.keras.Input(shape=(224,224,3))

    baseModel = tf.keras.applications.ResNet50(include_top=False, weights="imagenet", input_shape=(224,224,3), pooling="max")

    baseModel.trainable = trainable

    batchNormLayer = tf.keras.layers.BatchNormalization()

    denseLayer = tf.keras.layers.Dense(256, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(0.01))

    dropoutLayer = tf.keras.layers.Dropout(0.2)

    outputLayer = tf.keras.layers.Dense(80, activation="softmax")

    base = baseModel(input)
    batchNorm = batchNormLayer(base)
    dense = denseLayer(batchNorm)
    dropout = dropoutLayer(dense)
    output = outputLayer(dropout)

    model = tf.keras.Model(inputs=input, outputs=output)

    model.compile(optimizer=tf.keras.optimizers.Adamax(learning_rate=0.001),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    
    model.summary()

    return model

def resnet101_80():

    input = tf.keras.Input(shape=(224,224,3))

    baseModel = tf.keras.applications.ResNet101(include_top=False, weights="imagenet", input_shape=(224,224,3), pooling="max")

    batchNormLayer = tf.keras.layers.BatchNormalization()

    denseLayer = tf.keras.layers.Dense(256, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(0.01))

    dropoutLayer = tf.keras.layers.Dropout(0.2)

    outputLayer = tf.keras.layers.Dense(80, activation="softmax")

    base = baseModel(input)
    batchNorm = batchNormLayer(base)
    dense = denseLayer(batchNorm)
    dropout = dropoutLayer(dense)
    output = outputLayer(dropout)

    model = tf.keras.Model(inputs=input, outputs=output)

    model.compile(optimizer=tf.keras.optimizers.Adamax(learning_rate=0.001),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    
    model.summary()

    return model

def densenet121_80():

    input = tf.keras.Input(shape=(224,224,3))

    baseModel = tf.keras.applications.DenseNet121(include_top=False, weights="imagenet", input_shape=(224,224,3), pooling="max")

    batchNormLayer = tf.keras.layers.BatchNormalization()

    denseLayer = tf.keras.layers.Dense(256, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(0.01))

    dropoutLayer = tf.keras.layers.Dropout(0.2)

    outputLayer = tf.keras.layers.Dense(80, activation="softmax")

    base = baseModel(input)
    batchNorm = batchNormLayer(base)
    dense = denseLayer(batchNorm)
    dropout = dropoutLayer(dense)
    output = outputLayer(dropout)

    model = tf.keras.Model(inputs=input, outputs=output)

    model.compile(optimizer=tf.keras.optimizers.Adamax(learning_rate=0.001),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    
    model.summary()

    return model

def mobilenetV2_80(trainable = False):

    input = tf.keras.Input(shape=(224,224,3))

    baseModel = tf.keras.applications.MobileNetV2(include_top=False, weights="imagenet", input_shape=(224,224,3), pooling="max")

    baseModel.trainable = trainable

    batchNormLayer = tf.keras.layers.BatchNormalization()

    denseLayer = tf.keras.layers.Dense(256, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(0.01))

    dropoutLayer = tf.keras.layers.Dropout(0.2)

    outputLayer = tf.keras.layers.Dense(80, activation="softmax")

    base = baseModel(input)
    batchNorm = batchNormLayer(base)
    dense = denseLayer(batchNorm)
    dropout = dropoutLayer(dense)
    output = outputLayer(dropout)

    model = tf.keras.Model(inputs=input, outputs=output)

    model.compile(optimizer=tf.keras.optimizers.Adamax(learning_rate=0.001),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    
    model.summary()

    return model


def inceptionv3(trainable = False):

    """
    ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━┓
    ┃ Layer (type)                         ┃ Output Shape                ┃         Param # ┃
    ┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━┩
    │ input_layer (InputLayer)             │ (None, 224, 224, 3)         │               0 │
    ├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
    │ inception_v3 (Functional)            │ (None, 2048)                │      21,802,784 │
    ├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
    │ batch_normalization_94               │ (None, 2048)                │           8,192 │
    │ (BatchNormalization)                 │                             │                 │
    ├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
    │ dense (Dense)                        │ (None, 256)                 │         524,544 │
    ├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
    │ dropout (Dropout)                    │ (None, 256)                 │               0 │
    ├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
    │ dense_1 (Dense)                      │ (None, 80)                  │          20,560 │
    └──────────────────────────────────────┴─────────────────────────────┴─────────────────┘
    Total params: 22,356,080 (85.28 MB)
    Trainable params: 549,200 (2.10 MB)
    Non-trainable params: 21,806,880 (83.19 MB)
    
    """
    input = tf.keras.Input(shape=(224,224,3))

    baseModel = tf.keras.applications.InceptionV3(include_top=False, weights="imagenet", input_shape=(224,224,3), pooling="max")

    baseModel.trainable = trainable

    batchNormLayer = tf.keras.layers.BatchNormalization()

    denseLayer = tf.keras.layers.Dense(256, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(0.01))

    dropoutLayer = tf.keras.layers.Dropout(0.2)

    outputLayer = tf.keras.layers.Dense(80, activation="softmax")

    base = baseModel(input)
    batchNorm = batchNormLayer(base)
    dense = denseLayer(batchNorm)
    dropout = dropoutLayer(dense)
    output = outputLayer(dropout)

    model = tf.keras.Model(inputs=input, outputs=output)

    model.compile(optimizer=tf.keras.optimizers.Adamax(learning_rate=0.001),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    
    model.summary()

    return model


def inception_resnet_v2(trainable = False):
    input = tf.keras.Input(shape=(224,224,3))

    baseModel = tf.keras.applications.InceptionResNetV2(include_top=False, weights="imagenet", input_shape=(224,224,3), pooling="max")

    baseModel.trainable = trainable

    batchNormLayer = tf.keras.layers.BatchNormalization()

    denseLayer = tf.keras.layers.Dense(256, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(0.01))

    dropoutLayer = tf.keras.layers.Dropout(0.2)

    outputLayer = tf.keras.layers.Dense(80, activation="softmax")

    base = baseModel(input)
    batchNorm = batchNormLayer(base)
    dense = denseLayer(batchNorm)
    dropout = dropoutLayer(dense)
    output = outputLayer(dropout)

    model = tf.keras.Model(inputs=input, outputs=output)

    model.compile(optimizer=tf.keras.optimizers.Adamax(learning_rate=0.001),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    
    model.summary()

    return model


#TESTING MODEL

def testModel():
    trainval_ids, test_ids = sklearn.model_selection.train_test_split(np.arange(NUM_IMAGES), test_size=0.2)

    imageDirectory = "/Users/adamdavid/Desktop/Carleton University/4th Year/Winter '25/COMP 4107 A - Neural Networks/Project/aircraftClassification/images"

    labelsFilepath = "/Users/adamdavid/Desktop/Carleton University/4th Year/Winter '25/COMP 4107 A - Neural Networks/Project/aircraftClassification/labels.txt"

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


trainval_ids, test_ids = sklearn.model_selection.train_test_split(np.arange(NUM_IMAGES), test_size=0.2)

imageDirectory = "/Users/adamdavid/Desktop/Carleton University/4th Year/Winter '25/COMP 4107 A - Neural Networks/Project/aircraftClassification/images"

labelsFilepath = "/Users/adamdavid/Desktop/Carleton University/4th Year/Winter '25/COMP 4107 A - Neural Networks/Project/aircraftClassification/labels.txt"

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

model = tf.keras.models.load_model("/Users/adamdavid/Desktop/Carleton University/4th Year/Winter '25/COMP 4107 A - Neural Networks/Project/aircraftClassification/savedModels/fullyTrainable_BS_16/efficientnetb3_80_trainable_batch16.keras")


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

