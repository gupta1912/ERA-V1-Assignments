import albumentations as A
import matplotlib.pyplot as plt
import numpy as np
import torch
from albumentations.pytorch import ToTensorV2
from torchvision import datasets
from tqdm import tqdm


def get_dataset_stats(imgs):
    
    imgs = torch.stack(imgs, dim=0).numpy()

    mean_r = imgs[:,0,:,:].mean()
    mean_g = imgs[:,1,:,:].mean()
    mean_b = imgs[:,2,:,:].mean()
    mu = [mean_r,mean_g,mean_b]

    std_r = imgs[:,0,:,:].std()
    std_g = imgs[:,1,:,:].std()
    std_b = imgs[:,2,:,:].std()
    sigma = [std_r,std_g,std_b]
    return mu, sigma

def plot_grid(image, label, classes):

    nrows = 2
    ncols = 5

    fig, ax = plt.subplots(nrows, ncols, figsize=(8, 4))

    for i in range(nrows):
        for j in range(ncols):
            index = i * ncols + j
            ax[i, j].axis("off")
            ax[i, j].set_title("Label: %s" % (classes[label[index]]))
            ax[i, j].imshow(np.transpose(image[index], (1, 2, 0)))

class Cifar10Dataset(datasets.CIFAR10):

    def __init__(self, root="./data/cifar10", train=True, download=True, transform=None):
        super().__init__(root=root, train=train, download=download, transform=transform)

    def __getitem__(self, index):
        image, label = self.data[index], self.targets[index]

        if self.transform is not None:
            transformed = self.transform(image=image)
            image = transformed["image"]

        return image, label

def augmentation(data, mu, sigma):

    if data == 'Train':
        transform = A.Compose([A.HorizontalFlip(), A.ShiftScaleRotate(),
                                A.CoarseDropout(max_holes=1,
                                            max_height=16,
                                            max_width=16,
                                            min_holes=1,
                                            min_height=16,
                                            min_width=16,
                                            fill_value=np.mean(mu)),
                                A.ToGray(),
                                A.Normalize(mean=mu, std=sigma),
                                ToTensorV2()])
    else:
        transform = A.Compose([A.Normalize(mean=mu, std=sigma),
                                ToTensorV2()])

    return transform

def plot_curves(train_losses, train_acc, test_losses, test_acc):
    fig, axs = plt.subplots(1,2,figsize=(15,5))
    axs[0].plot(train_losses, label ='Train')
    axs[1].plot(train_acc, label ='Test')
    axs[0].plot(test_losses, label ='Train')
    axs[0].set_title("Loss")
    axs[1].plot(test_acc, label ='Test')
    axs[1].set_title("Accuracy")

def plot_misclassified(image, pred, target, classes):

    nrows = 2
    ncols = 5

    fig, ax = plt.subplots(nrows, ncols, figsize=(8, 4))

    for i in range(nrows):
        for j in range(ncols):
            index = i * ncols + j
            ax[i, j].axis("off")
            ax[i, j].set_title(f"Prediction: {classes[pred[index]]}\nTarget: {classes[target[index]]}")
            ax[i, j].imshow(image[index])

def GetCorrectPredCount(pPrediction, pLabels):
    return pPrediction.argmax(dim=1).eq(pLabels).sum().item()

def train(model, device, train_loader, optimizer, criterion, train_losses, train_acc):
    model.train()
    pbar = tqdm(train_loader)

    train_loss = 0
    correct = 0
    processed = 0

    for batch_idx, (data, target) in enumerate(pbar):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()

        # Predict
        pred = model(data)

        # Calculate loss
        loss = criterion(pred, target)
        train_loss+=loss.item()

        # Backpropagation
        loss.backward()
        optimizer.step()

        correct += GetCorrectPredCount(pred, target)
        processed += len(data)

        pbar.set_description(desc= f'Train: Loss={loss.item():0.4f} Batch_id={batch_idx} Accuracy={100*correct/processed:0.2f}')

    train_acc.append(100*correct/processed)
    train_losses.append(train_loss/len(train_loader))

    return train_losses, train_acc

def test(model, device, test_loader, criterion, test_losses, test_acc):
    model.eval()

    test_loss = 0
    correct = 0

    with torch.no_grad():
        for _, (data, target) in enumerate(test_loader):
            data, target = data.to(device), target.to(device)

            output = model(data)
            test_loss += criterion(output, target, reduction='sum').item()  # sum up batch loss

            correct += GetCorrectPredCount(output, target)


    test_loss /= len(test_loader.dataset)
    test_acc.append(100. * correct / len(test_loader.dataset))
    test_losses.append(test_loss)

    print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(test_loss, correct, len(test_loader.dataset),100. * correct / len(test_loader.dataset)))
    
    return test_losses, test_acc