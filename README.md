# EVA6-BatchNormalization-Regularization

Welcome, to learn more about implementation of Batch Normalization and Regularization using `Pytorch`, please continue reading patiently.

## Objective

1. Write a single model.py file that includes Group Normalization/Layer Normalization/Batch Normalization and takes an argument to decide which Normalization to include.  
2. Write a single notebook file to run all the 3 models above for 20 epochs each.    
3. Create these graphs:  
   Graph 1: Test/Validation Loss for all 3 models together.  
   Graph 2: Test/Validation Accuracy for 3 models together.  
   Graphs must have proper annotation.  
4. Find 10 misclassified images for each of the 3 models, and show them as a 5x2 image matrix in 3 separately annotated images.  

Lets begin!

Lets understand a bit about the 3 Normalizations that we have used, namely, **_Batch Normalization_**, _**Layer Normalization**_ and **_Group Normalization_**.

Consider the following setup

![image](https://user-images.githubusercontent.com/31658286/121711963-cc1bb480-caf8-11eb-86c8-d4bd3649dcf0.png)

We have two layers with batch size of 4, meaning 4 images in each batch.

![image](https://user-images.githubusercontent.com/31658286/121712318-40eeee80-caf9-11eb-874c-0c57ced2a6e8.png)

![image](https://user-images.githubusercontent.com/31658286/121712389-5401be80-caf9-11eb-8296-03472365ba00.png)

![image](https://user-images.githubusercontent.com/31658286/121712469-654acb00-caf9-11eb-83ef-7f91bd3248ec.png)



## MNIST Digit Recognition
Number of training samples: 60000  
Number of test samples: 10000  

## Transformations Used
1. Random Rotations  
2. Color Jitter  
3. Image Normalization  

## Normalization Techniques
1. Batch Normalization
2. Group Normalization
3. Layer Normalization

## Regularization
1. L1 Regularization  
Used Regularization factor of 0.0001  

## Findings
1. Model 1 - Group Normalization  
Train Accuracy: 99.63  
Test Accuracy: 99.52

2. Model 2 - Layer Normalization  
Train Accuracy: 99.58  
Test Accuracy: 99.46  

3. Model 3 - Batch Normalization + L1  
Train Accuracy: 99.48  
Test Accuracy: 99.48  


## Training and Validation - Loss & Accuracy
![train_test_loss_and_accuracy](https://user-images.githubusercontent.com/65554220/121549498-8b099e80-ca2b-11eb-9e13-9d4503b3cf94.png)


## Misclassified Images
1. Group Normalization  

![GN_missclassified](https://user-images.githubusercontent.com/65554220/121549846-d6bc4800-ca2b-11eb-99d9-1657f1d7ed51.png)


2. Layer Normalization  

![LN_missclassified](https://user-images.githubusercontent.com/65554220/121549886-df148300-ca2b-11eb-9fa8-afc3a7fd977a.png)

3. Batch Normalization + L1  

![BN_missclassified](https://user-images.githubusercontent.com/65554220/121549777-c86e2c00-ca2b-11eb-88f6-18c5f125f3aa.png)
