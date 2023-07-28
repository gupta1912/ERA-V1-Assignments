import torch
from torch.optim.lr_scheduler import OneCycleLR
from torch_lr_finder import LRFinder
from tqdm import tqdm


def get_dataloader(dataset, batch_size):
    dataloader_args = dict(shuffle=True, batch_size=batch_size, num_workers=4, pin_memory=True)
    loader = torch.utils.data.DataLoader(dataset, **dataloader_args)
    return loader

def find_lr(model, optimizer, criterion, dataloader, device):
    lr_finder = LRFinder(model, optimizer, criterion, device=device)
    lr_finder.range_test(dataloader, end_lr=10, num_iter=200, step_mode="exp")
    lr_finder.plot()
    suggested_lr = lr_finder.suggestion()
    print("Suggested Learning Rate:", suggested_lr) # to inspect the loss-learning rate graph
    lr_finder.reset()
    return suggested_lr

def get_scheduler(optimizer, train_loader, suggested_lr, EPOCHS):
    scheduler = OneCycleLR(
                            optimizer,
                            max_lr = suggested_lr,
                            steps_per_epoch=len(train_loader),
                            epochs=EPOCHS,
                            pct_start=5/EPOCHS,
                            div_factor=100,
                            three_phase=False,
                            final_div_factor=100,
                            anneal_strategy='linear'
                        )
    return scheduler

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


    test_loss /= len(test_loader)
    test_acc.append(100. * correct / len(test_loader.dataset))
    test_losses.append(test_loss)

    print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(test_loss, correct, len(test_loader.dataset),100. * correct / len(test_loader.dataset)))

    return test_losses, test_acc