# pytorchLearn
Notes for learning **machine learning** with **pytorch**
* Contains basic machine learning and mathematic knowledge.
* Contains pytorch usage cases and explanations 

## 1_tensor_basic.py
learn how to initialized tensor in pytorch

## 2_autograd.py
learn how grad mechanism is work in pytorch

## 3_Backpropagation
learning backpropagation in pytorch

## 4_n_Gradient_With_Autograd_and_Backpropagation.py
n setting are as followed
                            1       2           3               4
Prediction:             |Manual|Manual  |Manual           |PyTorch Model    |
Gradients Computation:  |Manual|Autograd|Autograd         |Autograd         |
Loss Computation:       |Manual|Manual  |PyTorch Loss     |PyTorch Loss     |
Parameter Updates:      |Manual|Manual  |PyTorch Optimizer|PyTorch Optimizer|

## 5_linear_regression.py
Linear Regression leaning in pytorch

## 6_logistic_regression.py
Learning logistics regression in pytorch

> logistics regression is similar with linear regression,
> it adds a logistics function(sigmoid) to convert it to a classification task

## 7_Dataset_Transforms
Learning how to load dataset using dataloader

## 8_Dataset_and_DataLoader
Dataset Transform in Pytorch

## 9_Softmax_and_Cross_Entropy
Learning Softmax and Cross Entropy in Pytorch

## 10_Activation_Function
Learning activation function and its' utility in pytorch torch.nn and torch.nn.functional

## 11_Feed_Forward_Neural_Network
Learning the DNN and let it learning on GPU

## 12_TensorBoard
Learning how to use TensorBoard to visualize training progress

## 13_Saving_and_Loading_Models
Learning ways to save and load models

## 14_RNN
Use rnn to do series processing task, use pytorch api of GRU and LSTM
(comparing to RNN and GRU, LSTM needs extra cell state c0)

## 15_learning_rate_adjust
Use pytorch to adjust learning rate to increase model performance

## Tutorial Reference
[Youtube(Pytorch Tutorials-Complete Beginner Course) Author.Python Engineer](https://www.youtube.com/playlist?list=PLqnslRFeH2UrcDBWF5mfPGpqQDSta6VK4)
