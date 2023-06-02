# Session 5 Assignment

Session 5 assignment is about learning how to modularize our code. This helps in keeping different objects required to train the model separate. Modular code increases the experimentation speed by many folds.

To practice, here we have taken MNIST Dataset and made a very simple Conolution network.

We ran this code in colab.

## Files & their descriptions

**[model.py](https://github.com/gupta1912/ERA-V1-Assignments/blob/main/S5/model.py)**

- This file defines a neural network model called __Net__ for image classification on the MNIST dataset.
- __Net__ class extends the __torch.nn.Module__ class and represents a convolutional neural network (CNN) architecture for image classification.

**[utils.py](https://github.com/gupta1912/ERA-V1-Assignments/blob/main/S5/utils.py)**

- This file provides utility functions for training and testing a model on the MNIST dataset using PyTorch.
- The __GetMNISTDataLoaders__ function creates and returns data loaders for the MNIST training and test datasets.
- The __train__ function performs training iterations on the model using the provided data loader, optimizer, and criterion.
- The __test__ function evaluates the trained model on the test dataset.
- The __TrainTestModel__ function combines the training and testing procedures for the model.

**[S5.ipynb](https://github.com/gupta1912/ERA-V1-Assignments/blob/main/S5/S5.ipynb) (notebook)**

- This is our working notebook. Here we import from all modules and run them as intended.
- We create dataloaders and model, train the model for 20 epochs and plot the loss and accuracy curves.

## Model Summary

Our toy CNN model summary is shown below:

```
----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Conv2d-1           [-1, 32, 26, 26]             288
            Conv2d-2           [-1, 64, 24, 24]          18,432
            Conv2d-3          [-1, 128, 10, 10]          73,728
            Conv2d-4            [-1, 256, 8, 8]         294,912
            Linear-5                   [-1, 50]         204,800
            Linear-6                   [-1, 10]             500
================================================================
Total params: 592,660
Trainable params: 592,660
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 0.00
Forward/backward pass size (MB): 0.67
Params size (MB): 2.26
Estimated Total Size (MB): 2.93
----------------------------------------------------------------
```
## Usage

Click on this button [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/gupta1912/ERA-V1-Assignments/). It will open the notebook in colab. Upload the [utils.py](https://github.com/gupta1912/ERA-V1-Assignments/blob/main/S5/utils.py) and [model.py](https://github.com/gupta1912/ERA-V1-Assignments/blob/main/S5/model.py) file in the files section. Change the notebook setting to __gpu__. Run the notebook cell wise.



