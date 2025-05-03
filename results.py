"""
Adam David
https://www.adamdavid.dev/

This file is used to plot results gotten from various 
experiments to allow us to visualize the data.
"""
#used to plot the results from running experiments

import matplotlib.pyplot as plt
import numpy as np

#Experiment on different base models

#Frozen, batchSize: 16, 5 epochs
def plotFrozenResults_b16():

    enetb3FrozenTrain = {'accuracy': [0.28252363204956055, 0.43709391355514526, 0.48065564036369324, 0.49870792031288147, 0.51229327917099], 'loss': [5.357443809509277, 3.1394870281219482, 2.708491563796997, 2.5709006786346436, 2.50091290473938]}
    enetb3FrozenTest = [2.3493943214416504, 0.563829779624939]

    resnet50FrozenTrain = {'accuracy': [0.3218030035495758, 0.5009598135948181, 0.5491361618041992, 0.5704370737075806, 0.5872341990470886], 'loss': [5.351900100708008, 2.9957151412963867, 2.5454142093658447, 2.405914545059204, 2.333890438079834]}
    resnet50FrozenTest = [2.3991293907165527, 0.5815602540969849]

    mobilenetv2FrozenTrain = {'accuracy': [0.09624187648296356, 0.16516537964344025, 0.19174541532993317, 0.20614294707775116, 0.2165534496307373], 'loss': [6.11132287979126, 4.195570468902588, 3.84137225151062, 3.7482986450195312, 3.6958703994750977]}
    mobilenetv2FrozenTest = [3.7983038425445557, 0.20656028389930725]

    inceptionv3FrozenTrain = {'accuracy': [0.24723124504089355, 0.3890652656555176, 0.43078115582466125, 0.4540756046772003, 0.46784552931785583], 'loss': [5.640341281890869, 3.3739075660705566, 2.9515492916107178, 2.8187179565429688, 2.7447218894958496]}
    inceptionv3FrozenTest = [2.849824905395508, 0.4552305042743683]

    inception_resnet_v2FrozenTrain = {'accuracy': [0.2217217981815338, 0.34598347544670105, 0.38533666729927063, 0.4063422977924347, 0.42000147700309753], 'loss': [5.597438812255859, 3.4476873874664307, 3.026602268218994, 2.8878302574157715, 2.822441577911377]}
    inception_resnet_v2FrozenTest = [2.8270304203033447, 0.4240543842315674]


    baseModels = ['Efficient Net B3', 'ResNet50', 'MobileNetV2', 'InceptionV3','InceptionResNetV2']
    epochs = [1,2,3,4,5]


    colors = {
        'InceptionResNetV2': 'red',
        'ResNet50' : 'orange',
        'InceptionV3': 'gold',
        'Efficient Net B3': 'green',
        'MobileNetV2': 'blue',
    }

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14,6))
    fig1, (ax3, ax4) = plt.subplots(1, 2, figsize=(14,6))


    #Training Accuracy and Loss

    for model in baseModels:

        
        if model == 'Efficient Net B3':
            accuracies = enetb3FrozenTrain['accuracy']
            losses = enetb3FrozenTrain['loss']

            tAcc = enetb3FrozenTest[1]
            tLoss = enetb3FrozenTest[0]

        elif model == 'ResNet50':
            accuracies = resnet50FrozenTrain['accuracy']
            losses = resnet50FrozenTrain['loss']

            tAcc = resnet50FrozenTest[1]
            tLoss = resnet50FrozenTest[0]

        elif model == 'MobileNetV2':
            accuracies = mobilenetv2FrozenTrain['accuracy']
            losses = mobilenetv2FrozenTrain['loss']

            tAcc = mobilenetv2FrozenTest[1]
            tLoss = mobilenetv2FrozenTest[0]

        elif model == 'InceptionV3':
            accuracies = inceptionv3FrozenTrain['accuracy']
            losses = inceptionv3FrozenTrain['loss']

            tAcc = inceptionv3FrozenTest[1]
            tLoss = inceptionv3FrozenTest[0]

        elif model == 'InceptionResNetV2':
            accuracies = inception_resnet_v2FrozenTrain['accuracy']
            losses = inception_resnet_v2FrozenTrain['loss']

            tAcc = inception_resnet_v2FrozenTest[1]
            tLoss = inception_resnet_v2FrozenTest[0]

        else:
            accuracies = []
            losses = []
            tAcc = 0
            tLoss = 0

        ax1.plot(epochs, accuracies, marker='o', color=colors[model], label=f'{model}')
        ax2.plot(epochs, losses, marker='x', color=colors[model], label=f'{model}')

        ax3.plot(np.array([5]), np.array([tAcc]), marker='o', color=colors[model], label=f'{model}')
        ax4.plot(np.array([5]), np.array([tLoss]), marker='x', color=colors[model], label=f'{model}')

    ax1.set_title('Training | Accuracy vs Epochs\nFrozen Base Models | Batch Size: 16')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Accuracy')
    ax1.legend()
    ax1.grid(True)

    ax2.set_title('Training | Loss vs Epochs\nFrozen Base Models | Batch Size: 16')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Loss')
    ax2.legend()
    ax2.grid(True)

    ax3.set_title('Testing | Accuracy\nFrozen Base Models | Batch Size: 16')
    ax3.set_xlabel('Epochs')
    ax3.set_ylabel('Accuracy')
    ax3.legend()
    ax3.grid(True)

    ax4.set_title('Testing | Loss\nFrozen Base Models | Batch Size: 16')
    ax4.set_xlabel('Epochs')
    ax4.set_ylabel('Loss')
    ax4.legend()
    ax4.grid(True)

    plt.show()

# plotFrozenResults_b16()

#Frozen, batchSize: 32, 5 epochs
def plotFrozenResults_b32():

    enetb3FrozenTrain = {'accuracy': [0.2704639434814453, 0.45209071040153503, 0.5134825706481934, 0.540669322013855, 0.5565528869628906], 'loss': [5.738407611846924, 3.3181750774383545, 2.6594722270965576, 2.4143693447113037, 2.3010175228118896]}
    enetb3FrozenTest = [2.2188870906829834, 0.5823459625244141]

    resnet50FrozenTrain = {'accuracy': [0.3463726341724396, 0.5674867033958435, 0.6333111524581909, 0.6653738021850586, 0.6846926808357239], 'loss': [5.546499729156494, 2.952805757522583, 2.2678639888763428, 2.0199573040008545, 1.9043052196502686]}
    resnet50FrozenTest = [2.125795364379883, 0.6218898296356201]

    mobilenetv2FrozenTrain = {'accuracy': [0.09681589901447296, 0.17305703461170197, 0.21102984249591827, 0.2355939745903015, 0.25147753953933716], 'loss': [6.471705436706543, 4.411917686462402, 3.8325212001800537, 3.6221866607666016, 3.5280723571777344]}
    mobilenetv2FrozenTest = [3.7594847679138184, 0.20497630536556244]

    inceptionv3FrozenTrain = {'accuracy': [0.24549350142478943, 0.4123817980289459, 0.4698212146759033, 0.4973773658275604, 0.5145907402038574], 'loss': [5.972947120666504, 3.5231683254241943, 2.8545567989349365, 2.615607738494873, 2.507328510284424]}
    inceptionv3FrozenTest = [2.737313985824585, 0.4748222827911377]

    inception_resnet_v2FrozenTrain = {'accuracy': [0.2137632966041565, 0.3572694957256317, 0.40998080372810364, 0.4387928545475006, 0.45685580372810364], 'loss': [5.9455485343933105, 3.645209312438965, 3.001108407974243, 2.75369930267334, 2.645435333251953]}
    inception_resnet_v2FrozenTest = [2.7073562145233154, 0.44771918654441833]


    baseModels = ['Efficient Net B3', 'ResNet50', 'MobileNetV2', 'InceptionV3','InceptionResNetV2']
    epochs = [1,2,3,4,5]


    colors = {
        'InceptionResNetV2': 'red',
        'ResNet50' : 'orange',
        'InceptionV3': 'gold',
        'Efficient Net B3': 'green',
        'MobileNetV2': 'blue',
    }

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14,6))
    fig1, (ax3, ax4) = plt.subplots(1, 2, figsize=(14,6))


    #Training Accuracy and Loss

    for model in baseModels:
        
        if model == 'Efficient Net B3':
            accuracies = enetb3FrozenTrain['accuracy']
            losses = enetb3FrozenTrain['loss']

            tAcc = enetb3FrozenTest[1]
            tLoss = enetb3FrozenTest[0]

        elif model == 'ResNet50':
            accuracies = resnet50FrozenTrain['accuracy']
            losses = resnet50FrozenTrain['loss']

            tAcc = resnet50FrozenTest[1]
            tLoss = resnet50FrozenTest[0]

        elif model == 'MobileNetV2':
            accuracies = mobilenetv2FrozenTrain['accuracy']
            losses = mobilenetv2FrozenTrain['loss']

            tAcc = mobilenetv2FrozenTest[1]
            tLoss = mobilenetv2FrozenTest[0]

        elif model == 'InceptionV3':
            accuracies = inceptionv3FrozenTrain['accuracy']
            losses = inceptionv3FrozenTrain['loss']

            tAcc = inceptionv3FrozenTest[1]
            tLoss = inceptionv3FrozenTest[0]

        elif model == 'InceptionResNetV2':
            accuracies = inception_resnet_v2FrozenTrain['accuracy']
            losses = inception_resnet_v2FrozenTrain['loss']

            tAcc = inception_resnet_v2FrozenTest[1]
            tLoss = inception_resnet_v2FrozenTest[0]

        else:
            accuracies = []
            losses = []
            tAcc = 0
            tLoss = 0

        ax1.plot(epochs, accuracies, marker='o', color=colors[model], label=f'{model}')
        ax2.plot(epochs, losses, marker='x', color=colors[model], label=f'{model}')

        ax3.plot(np.array([5]), np.array([tAcc]), marker='o', color=colors[model], label=f'{model}')
        ax4.plot(np.array([5]), np.array([tLoss]), marker='x', color=colors[model], label=f'{model}')

    ax1.set_title('Training | Accuracy vs Epochs\nFrozen Base Models | Batch Size: 32')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Accuracy')
    ax1.legend()
    ax1.grid(True)

    ax2.set_title('Training | Loss vs Epochs\nFrozen Base Models | Batch Size: 32')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Loss')
    ax2.legend()
    ax2.grid(True)

    ax3.set_title('Testing | Accuracy\nFrozen Base Models | Batch Size: 32')
    ax3.set_xlabel('Epochs')
    ax3.set_ylabel('Accuracy')
    ax3.legend()
    ax3.grid(True)

    ax4.set_title('Testing | Loss\nFrozen Base Models | Batch Size: 32')
    ax4.set_xlabel('Epochs')
    ax4.set_ylabel('Loss')
    ax4.legend()
    ax4.grid(True)

    plt.show()

# plotFrozenResults_b32()

#Frozen, batchSize: 32, 5 epochs
def plotFrozenResults_b64():

    enetb3FrozenTrain = {'accuracy': [0.25480201840400696, 0.44200649857521057, 0.5124483108520508, 0.553634762763977, 0.5756132006645203], 'loss': [6.14024019241333, 3.7126708030700684, 2.831320285797119, 2.4421370029449463, 2.2467195987701416]}
    enetb3FrozenTest = [2.207972526550293, 0.5925595164299011]

    resnet50FrozenTrain = {'accuracy': [0.32786643505096436, 0.5619089603424072, 0.6449837684631348, 0.6887928247451782, 0.7149084210395813], 'loss': [5.938431739807129, 3.3233587741851807, 2.40183162689209, 2.0081539154052734, 1.8097999095916748]}
    resnet50FrozenTest = [2.0494630336761475, 0.6532738208770752]

    # mobilenetv2FrozenTrain = {'accuracy': [0.09681589901447296, 0.17305703461170197, 0.21102984249591827, 0.2355939745903015, 0.25147753953933716], 'loss': [6.471705436706543, 4.411917686462402, 3.8325212001800537, 3.6221866607666016, 3.5280723571777344]}
    # mobilenetv2FrozenTest = [3.7594847679138184, 0.20497630536556244]

    # inceptionv3FrozenTrain = {'accuracy': [0.24549350142478943, 0.4123817980289459, 0.4698212146759033, 0.4973773658275604, 0.5145907402038574], 'loss': [5.972947120666504, 3.5231683254241943, 2.8545567989349365, 2.615607738494873, 2.507328510284424]}
    # inceptionv3FrozenTest = [2.737313985824585, 0.4748222827911377]

    # inception_resnet_v2FrozenTrain = {'accuracy': [0.2137632966041565, 0.3572694957256317, 0.40998080372810364, 0.4387928545475006, 0.45685580372810364], 'loss': [5.9455485343933105, 3.645209312438965, 3.001108407974243, 2.75369930267334, 2.645435333251953]}
    # inception_resnet_v2FrozenTest = [2.7073562145233154, 0.44771918654441833]


    baseModels = ['Efficient Net B3', 'ResNet50', 'MobileNetV2', 'InceptionV3','InceptionResNetV2']
    epochs = [1,2,3,4,5]


    colors = {
        'InceptionResNetV2': 'red',
        'ResNet50' : 'orange',
        'InceptionV3': 'gold',
        'Efficient Net B3': 'green',
        'MobileNetV2': 'blue',
    }

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14,6))
    fig1, (ax3, ax4) = plt.subplots(1, 2, figsize=(14,6))


    #Training Accuracy and Loss

    for model in baseModels:
        
        if model == 'Efficient Net B3':
            accuracies = enetb3FrozenTrain['accuracy']
            losses = enetb3FrozenTrain['loss']

            tAcc = enetb3FrozenTest[1]
            tLoss = enetb3FrozenTest[0]

        elif model == 'ResNet50':
            accuracies = resnet50FrozenTrain['accuracy']
            losses = resnet50FrozenTrain['loss']

            tAcc = resnet50FrozenTest[1]
            tLoss = resnet50FrozenTest[0]

        elif model == 'MobileNetV2':
            accuracies = mobilenetv2FrozenTrain['accuracy']
            losses = mobilenetv2FrozenTrain['loss']

            tAcc = mobilenetv2FrozenTest[1]
            tLoss = mobilenetv2FrozenTest[0]

        elif model == 'InceptionV3':
            accuracies = inceptionv3FrozenTrain['accuracy']
            losses = inceptionv3FrozenTrain['loss']

            tAcc = inceptionv3FrozenTest[1]
            tLoss = inceptionv3FrozenTest[0]

        elif model == 'InceptionResNetV2':
            accuracies = inception_resnet_v2FrozenTrain['accuracy']
            losses = inception_resnet_v2FrozenTrain['loss']

            tAcc = inception_resnet_v2FrozenTest[1]
            tLoss = inception_resnet_v2FrozenTest[0]

        else:
            accuracies = []
            losses = []
            tAcc = 0
            tLoss = 0

        ax1.plot(epochs, accuracies, marker='o', color=colors[model], label=f'{model}')
        ax2.plot(epochs, losses, marker='x', color=colors[model], label=f'{model}')

        ax3.plot(np.array([5]), np.array([tAcc]), marker='o', color=colors[model], label=f'{model}')
        ax4.plot(np.array([5]), np.array([tLoss]), marker='x', color=colors[model], label=f'{model}')

    ax1.set_title('Training | Accuracy vs Epochs\nFrozen Base Models | Batch Size: 32')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Accuracy')
    ax1.legend()
    ax1.grid(True)

    ax2.set_title('Training | Loss vs Epochs\nFrozen Base Models | Batch Size: 32')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Loss')
    ax2.legend()
    ax2.grid(True)

    ax3.set_title('Testing | Accuracy\nFrozen Base Models | Batch Size: 32')
    ax3.set_xlabel('Epochs')
    ax3.set_ylabel('Accuracy')
    ax3.legend()
    ax3.grid(True)

    ax4.set_title('Testing | Loss\nFrozen Base Models | Batch Size: 32')
    ax4.set_xlabel('Epochs')
    ax4.set_ylabel('Loss')
    ax4.legend()
    ax4.grid(True)

    plt.show()

# plotFrozenResults_b64()

#Frozen, plot testing accuracy and loss on different batch sizes, @ 5 epochs trained
def plotFrozenTestResults():

    #b16
    enetb3FrozenTest16 = [2.3493943214416504, 0.563829779624939]
    resnet50FrozenTest16 = [2.3991293907165527, 0.5815602540969849]
    mobilenetv2FrozenTest16 = [3.7983038425445557, 0.20656028389930725]
    inceptionv3FrozenTest16 = [2.849824905395508, 0.4552305042743683]
    inception_resnet_v2FrozenTest16 = [2.8270304203033447, 0.4240543842315674]


    #b32
    enetb3FrozenTest32 = [2.2188870906829834, 0.5823459625244141]
    resnet50FrozenTest32 = [2.125795364379883, 0.6218898296356201]
    mobilenetv2FrozenTest32 = [3.7594847679138184, 0.20497630536556244]
    inceptionv3FrozenTest32 = [2.737313985824585, 0.4748222827911377]
    inception_resnet_v2FrozenTest32 = [2.7073562145233154, 0.44771918654441833]



    baseModels = ['Efficient Net B3', 'ResNet50', 'MobileNetV2', 'InceptionV3','InceptionResNetV2']
    batchSizes = [16, 32]


    colors = {
        'InceptionResNetV2': 'red',
        'ResNet50' : 'orange',
        'InceptionV3': 'gold',
        'Efficient Net B3': 'green',
        'MobileNetV2': 'blue',
    }

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14,6))


    #Training Accuracy and Loss

    for model in baseModels:

        
        if model == 'Efficient Net B3':

            accuracies = [enetb3FrozenTest16[1],enetb3FrozenTest32[1]]
            losses = [enetb3FrozenTest16[0],enetb3FrozenTest32[0]]

        elif model == 'ResNet50':

            accuracies = [resnet50FrozenTest16[1],resnet50FrozenTest32[1]]
            losses = [resnet50FrozenTest16[0],resnet50FrozenTest32[0]]

        elif model == 'MobileNetV2':

            accuracies = [mobilenetv2FrozenTest16[1],mobilenetv2FrozenTest32[1]]
            losses = [mobilenetv2FrozenTest16[0],mobilenetv2FrozenTest32[0]]

        elif model == 'InceptionV3':

            accuracies = [inceptionv3FrozenTest16[1],inceptionv3FrozenTest32[1]]
            losses = [inceptionv3FrozenTest16[0],inceptionv3FrozenTest32[0]]

        elif model == 'InceptionResNetV2':

            accuracies = [inception_resnet_v2FrozenTest16[1],inception_resnet_v2FrozenTest32[1]]
            losses = [inception_resnet_v2FrozenTest16[0],inception_resnet_v2FrozenTest32[0]]

        else:
            accuracies = []
            losses = []

        ax1.plot(batchSizes, accuracies, marker='o', color=colors[model], label=f'{model}')
        ax2.plot(batchSizes, losses, marker='x', color=colors[model], label=f'{model}')

    ax1.set_title('Testing Accuracy vs Batch Size\nFrozen Base Models')
    ax1.set_xlabel('Batch Size')
    ax1.set_ylabel('Accuracy')
    ax1.legend()
    ax1.grid(True)

    ax2.set_title('Testing Loss vs Batch Size\nFrozen Base Models')
    ax2.set_xlabel('Batch Size')
    ax2.set_ylabel('Loss')
    ax2.legend()
    ax2.grid(True)

    plt.show()

# plotFrozenTestResults()

#Fully Trainable, batchSize: 16, 5 epochs
def plotResults1():
    """
    batch size: 16
    epochs: 5

    base models are fully trainable (backpropogate through the whole thing)
    with pre-processing
    """
    # string -> 
    testResults = {
        'Efficient Net B3':[0.6587037444114685, 0.882535457611084],
        'ResNet50': [1.5201414823532104, 0.7198581695556641],
        'MobileNetV2': [1.6070468425750732, 0.658687949180603],

    }
    results1 = {

        ('Efficient Net B3',1): (0.4467, 4.5344),
        ('Efficient Net B3',2): (0.7651, 1.5140),
        ('Efficient Net B3',3): (0.8684, 0.8237),
        ('Efficient Net B3',4): (0.9178, 0.5518),
        ('Efficient Net B3',5): (0.9454, 0.4001),

        ('Custom',1): (0.04946, 4.15),
        ('Custom',2): (0.05127, 4.09),
        ('Custom',3): (0.05098, 4.09),
        ('Custom',4): (0.05068, 4.09),
        ('Custom',5): (0.051609, 4.09),

        ('ResNet50', 1): (0.09786621481180191, 6.216208457946777),
        ('ResNet50', 2): (0.3244610130786896, 3.3619472980499268),
        ('ResNet50', 3): (0.5618724226951599, 2.2287347316741943),
        ('ResNet50', 4): (0.6958431601524353, 1.6653690338134766),
        ('ResNet50', 5): (0.7952967882156372, 1.244925856590271),

        ('MobileNetV2', 1): (0.2813422977924347, 5.127846717834473),
        ('MobileNetV2', 2): (0.5918487906455994, 2.183096408843994),
        ('MobileNetV2', 3): (0.7156305313110352, 1.4352290630340576),
        ('MobileNetV2', 4): (0.7947061657905579, 1.0779931545257568),
        ('MobileNetV2', 5): (0.835499107837677, 0.879355788230896),
        }

    types = ['Efficient Net B1', 'Efficient Net B3', 'ResNet50', 'ResNet101', 'DenseNet121', 'MobileNetV2', 'Custom']
    epochs = [1,2,3,4,5]

    # accuracies = [results1['Efficient Net B3', epoch][0] for epoch in epochs]
    # losses = [results1['Efficient Net B3', epoch][1] for epoch in epochs]

    colors = {
        'Efficient Net B1': 'red',
        'Efficient Net B3': 'green',
        'Custom' : 'orange',
        'ResNet50' : 'blue',
        'ResNet101' : 'purple',
        'DenseNet121' : 'brown',
        'MobileNetV2': 'yellow'
    }

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14,6))

    #Accuracy

    for type in types:
        accuracies = [results1[type, epoch][0] for epoch in epochs]
        ax1.plot(epochs, accuracies, marker='o', color=colors[type], label=f'{type}')

    ax1.set_title('Training | Accuracy vs Epochs')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Accuracy')
    ax1.legend()
    ax1.grid(True)

    for type in types:
        losses = [results1[type, epoch][1] for epoch in epochs]
        ax2.plot(epochs, losses, marker='x', color=colors[type], label=f'{type}')

    ax2.set_title('Training | Loss vs Epochs')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Loss')
    ax2.legend()
    ax2.grid(True)

    plt.show()
