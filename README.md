# Image Captioning Model with CNN and RNN

This repository contains the implementation of an **Image Captioning Model** as part of the **Pattern Recognition course (Assignment #3)** at **Alexandria University, Faculty of Engineering, Computer and Systems Engineering Department**. The project combines **Convolutional Neural Networks (CNN)** and **Recurrent Neural Networks (RNN)** to generate textual descriptions for images.

---

## Table of Contents
1. [Overview](#overview)
2. [Objective](#objective)
3. [Dataset](#dataset)
4. [Model Architecture](#model-architecture)
5. [Steps to Reproduce](#steps-to-reproduce)
6. [Results](#results)
7. [Future Enhancements](#future-enhancements)
8. [Resources](#resources)
9. [Acknowledgments](#acknowledgments)

---

## Overview

The objective of this project is to automatically generate textual captions for images by combining computer vision and natural language processing. **ResNet** is used for feature extraction, and **LSTM** is used for sequence generation. The model's performance is evaluated using **BLEU scores** on the **Flickr8k dataset**.
![Screenshot from 2024-11-19 04-34-47](https://github.com/user-attachments/assets/460ba513-d8c1-4d49-b3f8-6a2dca79bef6)


---

## Objective

The main goals of this project are:
- Apply CNN (ResNet) for image feature extraction.
- Utilize RNN (LSTM) for generating captions from extracted features.
- Learn and apply preprocessing techniques for image-captioning tasks.
- Evaluate model performance using BLEU scores and validation sets.
- Explore enhancements like hyperparameter tuning and attention mechanisms.

---

## Dataset

The project uses the **Flickr8k dataset**, which contains 8000 images, each with five captions. The dataset is split as follows:
- **Training set**: 6000 images
- **Validation set**: 1000 images
- **Test set**: 1000 images

### Dataset Links:
- [Flickr8k on Hugging Face](https://huggingface.co/datasets/jxie/flickr8k)
- [Flickr8k on Kaggle](https://www.kaggle.com/datasets/adityajn105/flickr8k/data)

---

## Model Architecture

### 1. Convolutional Neural Network (CNN)
- Pre-trained **ResNet** is used for feature extraction.
- The final layer of ResNet is frozen, and a fully connected layer is added for adaptation.

### 2. Recurrent Neural Network (RNN)
- **LSTM** is used to generate captions based on extracted features.
- **Teacher forcing** is applied during training to improve performance.
- The loss function used is **categorical cross-entropy**.

### 3. End-to-End Workflow
- **Input**: Images and captions.
- **CNN**: Extracts image features.
- **RNN**: Generates captions using the extracted features.

---

## Steps to Reproduce

### 1. Dataset Preparation
- Preprocess images by resizing and normalizing.
- Preprocess captions:
  - Tokenize captions into sequences.
  - Build a vocabulary and map words to unique indices.
  - Apply padding for uniform sequence lengths.

### 2. Feature Extraction using ResNet
- Load a pre-trained **ResNet** model and freeze its layers.
- Extract features from the layer before the classifier.

### 3. Caption Generation with LSTM
- Define an **LSTM-based RNN** model for caption generation.
- Train the model using teacher forcing and cross-entropy loss.

### 4. Training and Evaluation
- Train the model on the training set and validate using the validation set.
- Evaluate performance using BLEU scores on the test set.

### 5. Testing and Caption Generation
- Generate captions for test set images or downloaded images.
- Analyze where the model performs well or poorly.

---

## Results

- The model's performance is evaluated using **BLEU scores**.
- Example captions from the model are showcased in the notebook.
- Cases where the model underperforms are also analyzed and discussed.

---

## Future Enhancements

1. **Incorporate Attention Mechanisms**:
   - Focus on relevant parts of the image for each word in the caption.
   
2. **Hyperparameter Tuning**:
   - Experiment with learning rates, optimizers, and batch sizes.

3. **Improve Vocabulary**:
   - Consider training with all five captions per image for a more diverse vocabulary.

---

## Resources

- [Flickr8k Dataset on Hugging Face](https://huggingface.co/datasets/jxie/flickr8k)
- [Flickr8k Dataset on Kaggle](https://www.kaggle.com/datasets/adityajn105/flickr8k/data)
- [Medium Article: Image Caption Generator Using ResNet50 and LSTM](https://rupamgoyal12.medium.com/image-caption-generator-using-resnet50-and-lstm-model-a5b11f60cd23)

---

## Acknowledgments

This project was completed as part of the **Pattern Recognition course** under the guidance of:
- **Dr. Marwan Torki**
- **Eng. Ismail El Yamany**

Special thanks to Alexandria University, Faculty of Engineering, for providing this learning opportunity.
