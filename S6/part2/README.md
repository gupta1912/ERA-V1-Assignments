# Session 6 Assignment Part 2

This assignment is to learn how to optimize model and add difference layers to model. The objective was to get more than 99.4% accuracy in test/dev set for MNIST dataset such that model has less than 20,000 parameters and must achieve this in 20 epochs.

Ran this code in colab.

## Files & their descriptions

**[model.py](https://github.com/gupta1912/ERA-V1-Assignments/blob/main/S6/part2/model.py)**

- This file defines a neural network model called __Net__ for image classification on the MNIST dataset.
- __Net__ class extends the __torch.nn.Module__ class and represents a convolutional neural network (CNN) architecture for image classification.

**[utils.py](https://github.com/gupta1912/ERA-V1-Assignments/blob/main/S6/part2/utils.py)**

- This file provides utility functions for training and testing a model on the MNIST dataset using PyTorch.
- The __train__ function performs training iterations on the model using the provided data loader, optimizer, and criterion.
- The __test__ function evaluates the trained model on the test dataset.

**[S6_part_2.ipynb](https://github.com/gupta1912/ERA-V1-Assignments/blob/main/S6/part2/S6_part_2.ipynb) (notebook)**

- This is our working notebook. Here we import from all modules and run them as intended.
- We create dataloaders and model, train the model for 20 epochs.

## Model Summary

Our CNN model summary is shown below:

```
----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Conv2d-1            [-1, 8, 26, 26]              80
       BatchNorm2d-2            [-1, 8, 26, 26]              16
           Dropout-3            [-1, 8, 26, 26]               0
            Conv2d-4           [-1, 12, 24, 24]             876
       BatchNorm2d-5           [-1, 12, 24, 24]              24
           Dropout-6           [-1, 12, 24, 24]               0
            Conv2d-7           [-1, 16, 22, 22]           1,744
       BatchNorm2d-8           [-1, 16, 22, 22]              32
           Dropout-9           [-1, 16, 22, 22]               0
        MaxPool2d-10           [-1, 16, 11, 11]               0
           Conv2d-11            [-1, 8, 11, 11]             136
      BatchNorm2d-12            [-1, 8, 11, 11]              16
          Dropout-13            [-1, 8, 11, 11]               0
           Conv2d-14             [-1, 12, 9, 9]             876
      BatchNorm2d-15             [-1, 12, 9, 9]              24
          Dropout-16             [-1, 12, 9, 9]               0
           Conv2d-17             [-1, 16, 7, 7]           1,744
      BatchNorm2d-18             [-1, 16, 7, 7]              32
          Dropout-19             [-1, 16, 7, 7]               0
           Conv2d-20             [-1, 16, 5, 5]           2,320
      BatchNorm2d-21             [-1, 16, 5, 5]              32
          Dropout-22             [-1, 16, 5, 5]               0
           Conv2d-23             [-1, 16, 3, 3]           2,320
      BatchNorm2d-24             [-1, 16, 3, 3]              32
          Dropout-25             [-1, 16, 3, 3]               0
           Conv2d-26             [-1, 10, 3, 3]             170
        AvgPool2d-27             [-1, 10, 1, 1]               0
================================================================
Total params: 10,474
Trainable params: 10,474
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 0.00
Forward/backward pass size (MB): 0.55
Params size (MB): 0.04
Estimated Total Size (MB): 0.59
----------------------------------------------------------------
```
## Usage

Click on this button [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/gupta1912/ERA-V1-Assignments/). It will open the notebook in colab. Upload the [utils.py](https://github.com/gupta1912/ERA-V1-Assignments/blob/main/S6/part2/utils.py) and [model.py](https://github.com/gupta1912/ERA-V1-Assignments/blob/main/S6/part2/model.py) file in the files section. Change the notebook setting to __gpu__. Run the notebook cell wise.

## Result

Got 99.43% accuracy on test set at epoch 12. Total number of parameters in the model are 10,474.



