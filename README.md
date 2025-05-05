# Military Aircraft Image Classification  
**COMP 4107 - Neural Networks Final Project**  
**Author:** [Adam David](https://www.adamdavid.dev)  


---

## Overview
The goal of this project is to classify images of military aircraft into 80 distinct classes using deep learning. This project leverages CNN-based architectures such as EfficientNetB3, ResNet50, and MobileNetV2 as base models due to their high accuracy on the ImageNet dataset. This project uses the dataset from Kaggle's *Military Aircraft Detection Dataset*, and all models are built using Tensorflow/Keras. The models are trained both with the base models as frozen feature extractors and while the base models are fully trainable. 
<!--
Due to hardware limitations (no access to a GPU), the object detection component was not implemented, but future work is planned to incorporate YOLO or EfficientDet for localization.
-->
---

## Libraries Used

  - **OS/Shutil**: Reformat the dataset

  - **Tensorflow/Keras**: Build, train and test various models

  - **Pillow/Numpy**: Load, generate, and pre-process data
    
  - **MatPlotLib**: Create visualizations for results data

---

---

## Project Structure

```
military-aircraft-classification/
├── images/                                            # Reformatted and structured dataset
│   └── 33,872 .jpg images                             # Images
├── labels.txt/                                        # Class label mappings
├── savedModels/                                       # Saved model weights and architectures
│   └── various saved models
├── results/                                           # Accuracy/Loss plots
│   ├── batchSize16_frozen_training_accuracy_loss.png
│   ├── batchSize32_frozen_training_accuracy_loss.png
│   └── batchSize_vs_frozen_test_accuracy_loss_.png
├── README.md                                          # Project overview and documentation
├── classification.py                                  # Main script, contains training, evaluation, data generator, and architecture definitions
├── results.py                                         # Results script, used to visualize the results
└── requirements.txt                                   # Python dependencies
```
---

## Dataset  

**Source:** [Military Aircraft Detection Dataset (Kaggle)](https://www.kaggle.com/datasets/a2015003713/militaryaircraftdetectiondataset/data)  

### Classification Set:
- **Total images:** 33,872  
- **Classes:** 80 types of military aircraft  
- **Image sizes:** Ranges from 5x10 px to 4912x7360 px  
- **Preprocessing:** All images resized to 224x224 for model compatibility  
- **Split:** 80% training, 20% testing  

### Object Detection Set:
- **Total images:** 19,214  
- **Annotations:** CSV files with bounding box coordinates and class labels  
- **Split:** 75% training, 18% validation, 7% testing  

---

## Models and Methods

### Classification
- **Pre-trained Base Models:**  
  - EfficientNetB3  
  - ResNet50  
  - MobileNetV2  
  - InceptionV3  
  - InceptionResNetV2  

- **Architecture:**
  - Pre-trained base (frozen/unfrozen)
  - Batch Normalization Layer
  - **Dense Layer:** 256 neurons, ReLU activation, L2 regularization
  - **Dropout Layer:** 20% disabled
  - **Output Layer**: 80 neurons, softmax activation (for multi-class classification)

- **Training Details:**
  - Input size: 224x224x3  
  - Optimizer: **Adamax** (a variant of Adam based on the infinity norm) 
  - Loss: **Categorical Cross-Entropy**  
  - Evaluation metrics: Accuracy & Loss over epochs  

- Batch Sizes Tested: **16 and 32**
---

## 📊 Results of unfrozen EfficientNetB3 trained for 14 epochs
- Training Accuracy: **98.5%**
- Training Loss: **0.14**
- Testing Accuracy: **95.5%**
- Testing Loss: **0.265**

## 📊 Results of *unfrozen* models trained for 5 epochs

| Model             | Train Acc  | Test Acc | Batch Size | Notes                        |
|------------------|----------------------|----------|------------|------------------------------|
| EfficientNetB3    | 95%                  | 88%        | 16         | Best performance    |
| ResNet50          | 80%                  | 72%      | 16         |          |
| MobileNetV2       | 83%                  | 65%      | 16         | Fastest but lowest accuracy  |


## 📊 Results of *frozen* models trained for 5 epochs

**Training**
![alt text](https://github.com/adxmd/military-aircraft-classification/blob/main/results/batchSize16_frozen_training_accuracy_loss.png?raw=true)


![alt text](https://github.com/adxmd/military-aircraft-classification/blob/main/results/batchSize32_frozen_training_accuracy_loss.png?raw=true)

**Testing**

![alt text](https://github.com/adxmd/military-aircraft-classification/blob/main/results/batchSize_vs_frozen_test_accuracy_loss_.png?raw=true)

**Notes:**
- We can see improvements in model performance if trained for longer
- ResNet50 performed the best when used as a frozen feature extractor, probably due to its residual connections and balanced depth.
- MobileNetV2 is too lightweight to extract enough features when frozen, hence its low accuracy
- InceptionResNetV2's lack of accuracy when frozen proves that it is too complex, with more than double the parameters of any other model here
- Increasing the batch size increased accuracy and lowered categorical cross-entropy loss across most models

---

## ⚠ Limitations
- CPU-only training significantly increased runtime and limited hyperparameter exploration.
- Object detection was not implemented due to the lack of CUDA support.
- Very small input images suffered heavily from information loss during resizing.

---

## Future Work
- Implement object detection using YOLO, EfficientDet, or Faster R-CNN.
- Use GPU-accelerated training to fine-tune full models and explore more hyperparameters.


---

## 🔗 References  
- [EfficientNet Paper](https://arxiv.org/abs/1905.11946)  
- [ResNet Paper](https://arxiv.org/abs/1512.03385)  
- [MobileNet Paper](https://arxiv.org/abs/1704.04861)  

