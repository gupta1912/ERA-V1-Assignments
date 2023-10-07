# Session 18 - ERA Phase I - Assignment 

## Objective

1. UNet - Train with the below four variations of architecture/loss

    MP+Tr+BCE
    
    MP+Tr+Dice Loss
    
    StrConv+Tr+BCE
    
    StrConv+Ups+Dice Loss
    
2. VAE - For the following dataset customize to have an input (image and label)

    MNIST
    
    CIFAR10


## Result

1. Example of prediction using UNet 
![results](./results/eg_UNet.png)

    A. MP+Tr+BCE
![results](./results/UNet_MP_Tr_BCE.png)

    B. MP+Tr+Dice Loss
![results](./results/UNet_MP_Tr_DICE.png)

    C. StrConv+Tr+BCE
![results](./results/UNet_Stride_Tr_BCE.png)

    D. StrConv+Ups+Dice Loss
![results](./results/UNet_Stride_Ups_DICE.png)

2. VAE

    A. MNIST --> Input image + 25 different iterations of wrong label
![results](./results/VAE_MNIST.png)
    B. CIFAR10 --> Input image + 25 different iterations of wrong label
![results](./results/VAE_CIFAR10.png)



Contributors
-------------------------
Lavanya Nemani 

Shashank Gupta 