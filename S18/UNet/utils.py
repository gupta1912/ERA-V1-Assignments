import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm


def train_unet(model, device, train_loader, optimizer, train_losses, criterion):
    model.train()
    pbar = tqdm(train_loader)
    train_loss = 0

    for _, (data, target, label_one) in enumerate(pbar):
        # Convert label_one to the appropriate data type
        label_one = label_one.to(torch.float32)
        
        data, target, label_one = data.to(device), target.to(device), label_one.to(device)

        # Zero out gradients
        optimizer.zero_grad()

        # Forward pass
        y_pred = model(data)

        # Calculate loss
        loss = criterion(y_pred, label_one)

        train_loss += loss.item()

        # Backpropagation and optimization
        loss.backward()
        optimizer.step()

    train_losses.append(train_loss)

    print(f'Training Loss: {train_loss}')
    return train_losses


def test_unet(model, device, test_loader,test_losses,criterion):
    model.eval()
    test_loss = 0

    with torch.no_grad():
        for data, target, label_one in test_loader:
            label_one = label_one.to(torch.float32)
            data, target, label_one = data.to(device), target.to(device), label_one.to(device)
            output = model(data)
            loss = criterion(output, label_one)
            test_loss += loss.item()


    #test_loss /= len(test_loader.dataset)
    test_losses.append(test_loss)

    print(f'Test set: Average loss={test_loss}')
    return test_losses
    

def fit_model_unet(model, optimizer, criterion, trainloader, testloader, EPOCHS, device):
    train_losses = []
    test_losses = []
    
    for epoch in range(EPOCHS):
        print("\n EPOCH: {} (LR: {})".format(epoch+1, optimizer.param_groups[0]['lr']))
        train_losses = train_unet(model, device, trainloader, optimizer, train_losses, criterion)
        test_losses = test_unet(model, device, testloader, test_losses, criterion)

    return model, train_losses, test_losses

# for binary class
def unet_dice_loss(pred, target):
    smooth = 1e-5

    # flatten predictions and targets
    pred = pred.view(-1)
    target = target.view(-1)
    
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum()
    
    dice = (2. * intersection + smooth) / (union + smooth)
    
    return 1 - dice   

def show_sample(dataset, num_samples=4):
    data_iterator = iter(dataset)

    _, axes = plt.subplots(2, num_samples, figsize=(20, 10))
    
    for i in range(num_samples):
        images, labels, _ = next(data_iterator)
        
        # Plot Original Image
        axes[0, i].set_title(f'\n Label: Original', fontsize=6)
        axes[0, i].imshow(np.transpose(images, (1, 2, 0)))
        
        # Plot Ground Truth
        axes[1, i].set_title(f'\n Label: Ground Truth', fontsize=6)
        axes[1, i].imshow(labels)

    plt.tight_layout()
    plt.show()

def show_sample_output(model, loader, device, image_no=2):
    dataiter = iter(loader)

    with torch.no_grad():
        _, axes = plt.subplots(image_no, 3, figsize=(10, 10))
        for i in range(image_no):
            images, labels, _ = next(dataiter)
            images, labels = images.to(device), labels.to(device)

            # Original Image
            ax = axes[i, 0]
            ax.set_title('Original', fontsize=6)
            ax.imshow(np.transpose(images[0].cpu().numpy(), (1, 2, 0)))

            # Ground Truth
            ax = axes[i, 1]
            ax.set_title('Ground Truth', fontsize=6)
            ax.imshow(labels[0].cpu().numpy())

            # Predicted Mask
            output = model(images).squeeze()
            predicted_masks = torch.argmax(output, 1).cpu().numpy()
            ax = axes[i, 2]
            ax.set_title('Predicted', fontsize=6)
            ax.imshow(predicted_masks[0])

    plt.tight_layout()
    plt.show()

def plot_curves(train_losses, test_losses):
    _, axs = plt.subplots(1,1,figsize=(10,5))
    axs.plot(train_losses, label ='Train')
    axs.plot(test_losses, label ='Test')
    axs.set_title("Loss")
    plt.show()
