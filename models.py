import tensorflow as tf
from tensorflow import keras


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

