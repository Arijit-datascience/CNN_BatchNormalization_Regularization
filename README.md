# EVA6-BatchNormalization-Regularization

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
1. Model 1 - Batch Normalization + L1  
Train Accuracy: 99.48  
Test Accuracy: 99.48  

2. Model 2 - Group Normalization  
Train Accuracy: 99.63  
Test Accuracy: 99.52

3. Model 3 - Layer Normalization  
Train Accuracy: 99.58  
Test Accuracy: 99.46  


## Training and Validation - Loss & Accuracy
![train_test_loss_and_accuracy](https://user-images.githubusercontent.com/65554220/121549498-8b099e80-ca2b-11eb-9e13-9d4503b3cf94.png)


## Misclassified Images
1. Batch Normalization + L1  

![BN_missclassified](https://user-images.githubusercontent.com/65554220/121549777-c86e2c00-ca2b-11eb-88f6-18c5f125f3aa.png)


2. Group Normalization  

![GN_missclassified](https://user-images.githubusercontent.com/65554220/121549846-d6bc4800-ca2b-11eb-99d9-1657f1d7ed51.png)


3. Layer Normalization  

![LN_missclassified](https://user-images.githubusercontent.com/65554220/121549886-df148300-ca2b-11eb-9fa8-afc3a7fd977a.png)
