import torch
from torchvision import datasets
from tqdm import tqdm


def GetMNISTDataLoaders(train_transforms, test_transforms, data_dir = './data', batch_size=512, num_workers=2, pin_memory=True):
    train_data = datasets.MNIST(data_dir, train=True, download=True, transform=train_transforms)
    test_data = datasets.MNIST(data_dir, train=False, download=True, transform=test_transforms)
    kwargs = {'batch_size': batch_size, 'shuffle': True, 'num_workers': num_workers, 'pin_memory': pin_memory}

    test_loader = torch.utils.data.DataLoader(test_data, **kwargs)
    train_loader = torch.utils.data.DataLoader(train_data, **kwargs)

    return train_loader, test_loader

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

def TrainTestModel(model, criterion, optimizer, scheduler, train_loader, test_loader, num_epochs, device):
    train_losses = []
    test_losses = []
    train_acc = []
    test_acc = []

    for epoch in range(1, num_epochs+1):
        print(f'Epoch {epoch}')
        train_losses, train_acc = train(model, device, train_loader, optimizer, criterion, train_losses, train_acc)
        test_losses, test_acc = test(model, device, test_loader, criterion, test_losses, test_acc)
        scheduler.step()

    return train_losses, train_acc, test_losses, test_acc