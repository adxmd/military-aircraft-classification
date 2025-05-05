# Military Aircraft Image Classification  
**COMP 4107 - Neural Networks Final Project**  
**Author:** Adam David  


---

## Overview
The goal of this project is to classify images of military aircraft into 80 distinct classes using deep learning. I leverage CNN-based architectures such as EfficientNetB3, ResNet50, and MobileNetV2 as base models due to their high accuracy and efficiency when pre-trained on the ImageNet dataset. I will be using the dataset from Kaggle's *Military Aircraft Detection Dataset*, and all models are built using Tensorflow/Keras.
<!--
Due to hardware limitations (no access to a GPU), the object detection component was not implemented, but future work is planned to incorporate YOLO or EfficientDet for localization.
-->
---

## 📂 Dataset  

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

## 🧠 Models and Methods

### ✅ Classification
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
  - Loss: **Categorical Crossentropy**  
  - Evaluation metrics: Accuracy & Loss over epochs  

- **Batch Sizes Tested:** 16 and 32  

---
