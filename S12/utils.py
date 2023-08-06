# imports
import albumentations as A
import lightning as L
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.optim as optim
from albumentations.pytorch import ToTensorV2
from model import MyResNet
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import preprocess_image, show_cam_on_image
from torch import nn
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import DataLoader
from torchmetrics.functional import accuracy
from torchvision import datasets, transforms

means = [0.4914, 0.4822, 0.4465]
stds = [0.2470, 0.2435, 0.2616]

class CustomResnetTransforms:
    def train_transforms(means, stds):
        return A.Compose(
                [
                    A.Normalize(mean=means, std=stds, always_apply=True),
                    A.PadIfNeeded(min_height=36, min_width=36, always_apply=True),
                    A.RandomCrop(height=32, width=32, always_apply=True),
                    A.HorizontalFlip(),
                    A.Cutout(num_holes=1, max_h_size=8, max_w_size=8, fill_value=0, p=1.0),
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
    
class Cifar10SearchDataset(datasets.CIFAR10):

    def __init__(self, root="~/data", train=True, download=True, transform=None):
        super().__init__(root=root, train=train, download=download, transform=transform)

    def __getitem__(self, index):
        image, label = self.data[index], self.targets[index]
        if self.transform is not None:
            transformed = self.transform(image=image)
            image = transformed["image"]
        return image, label
    
class LitCIFAR10(L.LightningModule):
    def __init__(self, data_dir='./data', learning_rate=0.01, batch_size = 512):

        super().__init__()

        # Set our init args as class attributes
        self.data_dir = data_dir
        self.lr = learning_rate
        self.batch_size = batch_size

        # Hardcode some dataset specific attributes
        self.num_classes = 10
        self.train_transforms = CustomResnetTransforms.train_transforms(means, stds)
        self.test_transforms = CustomResnetTransforms.test_transforms(means, stds)

        # Define PyTorch model
        self.model = MyResNet()
        self.criterion = nn.CrossEntropyLoss()


    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        preds = torch.argmax(logits, dim=1)
        acc = accuracy(preds, y, task='multiclass',
                                     num_classes=10)

        # Calling self.log will surface up scalars for you in TensorBoard

        self.log("train_loss", loss, prog_bar=True, enable_graph = True, on_step=False, on_epoch=True)
        self.log("train_acc", acc, prog_bar=True, enable_graph = True, on_step=False, on_epoch=True)
        # print("train_loss", loss)
        # print("train_acc", acc)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        preds = torch.argmax(logits, dim=1)
        acc = accuracy(preds, y, task='multiclass',
                                     num_classes=10)

        # Calling self.log will surface up scalars for you in TensorBoard
        self.log("val_loss", loss, prog_bar=True, enable_graph = True, on_step=False, on_epoch=True)
        self.log("val_acc", acc, prog_bar=True, enable_graph = True, on_step=False, on_epoch=True)
        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.lr, weight_decay=1e-4)

        steps_per_epoch = (len(self.trainset) // self.batch_size)+1
        scheduler_dict = {
            "scheduler": OneCycleLR(
                                    optimizer,
                                    max_lr = self.lr,
                                    steps_per_epoch=steps_per_epoch,
                                    epochs=self.trainer.max_epochs,
                                    pct_start=5/self.trainer.max_epochs,
                                    div_factor=100,
                                    three_phase=False,
                                    final_div_factor=100,
                                    anneal_strategy='linear'
                                ),
            "interval": "step",
        }
        return {"optimizer": optimizer, "lr_scheduler": scheduler_dict}

    def setup(self, stage=None):

        # Assign train/val datasets for use in dataloaders
        self.trainset = Cifar10SearchDataset(root=self.data_dir, train=True,
                                    download=True, transform=self.train_transforms)
        self.valset = Cifar10SearchDataset(root=self.data_dir, train=False,
                                    download=True, transform=self.test_transforms)

    def train_dataloader(self):
        return DataLoader(self.trainset, batch_size=self.batch_size, num_workers=0, pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.valset, batch_size=self.batch_size, num_workers=0, pin_memory=True)

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
            target_layers = [model.model.layer3[-1]]
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