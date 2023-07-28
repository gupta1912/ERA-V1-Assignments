import warnings

warnings.filterwarnings('ignore')
warnings.simplefilter('ignore')

import albumentations as A
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from albumentations.pytorch import ToTensorV2
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import preprocess_image, show_cam_on_image
from torchvision import datasets


def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

def visualise_train_images(train_loader, classes):
    # get some random training images
    dataiter = iter(train_loader)
    images, labels = next(dataiter)
    # show images
    imshow(torchvision.utils.make_grid(images[:4]))
    # print labels
    print(' '.join(f'{classes[labels[j]]:5s}' for j in range(4)))

class CustomResnetTransforms:
    def train_transforms(means, stds):
        return A.Compose(
                [
                    A.Normalize(mean=means, std=stds, always_apply=True),
                    A.PadIfNeeded(min_height=36, min_width=36, always_apply=True),
                    A.RandomCrop(height=32, width=32, always_apply=True),
                    A.Cutout(num_holes=1, max_h_size=16, max_w_size=16, fill_value=0, p=1.0),
                    ToTensorV2(),
                ]
            )
    
    def test_transforms(means, stds):
         return A.Compose(
            [
                A.Normalize(mean=means, std=stds, always_apply=True),
                ToTensorV2(),
            ]
        )

def plot_curves(train_losses, train_acc, test_losses, test_acc):
    fig, axs = plt.subplots(1,2,figsize=(15,5))
    axs[0].plot(train_losses, label ='Train')
    axs[1].plot(train_acc, label ='Test')
    axs[0].plot(test_losses, label ='Train')
    axs[0].set_title("Loss")
    axs[1].plot(test_acc, label ='Test')
    axs[1].set_title("Accuracy")

def get_misclassified_images(model, testset, mu, sigma, device):
    model.eval()
    transform=transforms.Compose([
                            transforms.ToTensor(),
                            transforms.Normalize(mu, sigma)
                        ])
    misclassified_images, misclassified_predictions, true_targets = [], [], []
    with torch.no_grad():
        for data_, target in testset:
            data = transform(data_).to(device)
            data = data.unsqueeze(0)
            output = model(data)
            pred = output.argmax(dim=1, keepdim=True)

            if pred.item()!=target:
                misclassified_images.append(data_)
                misclassified_predictions.append(pred.item())
                true_targets.append(target)
    return misclassified_images, misclassified_predictions, true_targets

def plot_misclassified(image, pred, target, classes):

    nrows = 4
    ncols = 5

    _, ax = plt.subplots(nrows, ncols, figsize=(20, 15))

    for i in range(nrows):
        for j in range(ncols):
            index = i * ncols + j
            ax[i, j].axis("off")
            ax[i, j].set_title(f"Prediction: {classes[pred[index]]}\nTarget: {classes[target[index]]}")
            ax[i, j].imshow(image[index])
    plt.show()


class ClassifierOutputTarget:
    def __init__(self, category):
        self.category = category

    def __call__(self, model_output):
        if len(model_output.shape) == 1:
            return model_output[self.category]
        return model_output[:, self.category]

def plot_grad_cam_images(images, pred, target, classes, model):
    nrows = 4
    ncols = 5
    fig, ax = plt.subplots(nrows, ncols, figsize=(20,15))
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    for i in range(nrows):
        for j in range(ncols):
            index = i * ncols + j
            img = images[index]
            input_tensor = preprocess_image(img,
                                mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])
            target_layers = [model.layer4[-1]]
            targets = [ClassifierOutputTarget(target[index])]
            cam = GradCAM(model=model, target_layers=target_layers, use_cuda = device)
            grayscale_cam = cam(input_tensor=input_tensor, targets = targets)
            #grayscale_cam = cam(input_tensor=input_tensor)
            grayscale_cam = grayscale_cam[0, :]
            rgb_img = np.float32(img) / 255
            visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True, image_weight = 0.6)

            index = i * ncols + j
            ax[i, j].axis("off")
            ax[i, j].set_title(f"Prediction: {classes[pred[index]]}\nTarget: {classes[target[index]]}")
            ax[i, j].imshow(visualization)
    plt.show()


