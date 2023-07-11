import albumentations as A
import matplotlib.pyplot as plt
import numpy as np
import torch
from albumentations.pytorch import ToTensorV2
from torchinfo import summary
from tqdm import tqdm


def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

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
    
def plot_curves(train_losses, train_acc, test_losses, test_acc):
    fig, axs = plt.subplots(1,2,figsize=(15,5))
    axs[0].plot(train_losses, label ='Train')
    axs[1].plot(train_acc, label ='Test')
    axs[0].plot(test_losses, label ='Train')
    axs[0].set_title("Loss")
    axs[1].plot(test_acc, label ='Test')
    axs[1].set_title("Accuracy")

def GetCorrectPredCount(pPrediction, pLabels):
    return pPrediction.argmax(dim=1).eq(pLabels).sum().item()

def train(model, device, train_loader, optimizer, criterion, scheduler, train_losses, train_acc):
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
        scheduler.step()

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
            test_loss += criterion(output, target).item()  # sum up batch loss

            correct += GetCorrectPredCount(output, target)


    test_loss /= len(test_loader.dataset)
    test_acc.append(100. * correct / len(test_loader.dataset))
    test_losses.append(test_loss)

    print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(test_loss, correct, len(test_loader.dataset),100. * correct / len(test_loader.dataset)))

    return test_losses, test_acc
    
