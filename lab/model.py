import os
import torch
import torch.nn as nn
import torchvision
import pandas as pd

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

def get_net(net_name, pretrained=True):
    """
    Get network architecture from name

    Args:
        net_name (string): name of the network
        pretrained (bool): used pre-trained weights or random weights

    Rets:
        torch.nn.Module
    """
    if net_name == 'resnet18_imagenet':
        net = torchvision.models.resnet18(pretrained=pretrained)
    if net_name == 'resnet18_cifar':
        net = torchvision.models.resnet18()
    if net_name == 'cnn_mnist':
        net = nn.Sequential(
            nn.Conv2d(1, 64, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 4, stride=2, padding=1),
            nn.ReLU(),
            Flatten(),
            nn.Linear(128*7*7,500),
            nn.ReLU(),
            nn.Linear(500,100),
            nn.ReLU(),
            nn.Linear(100, 10)
        )
    return net

class Model():
    def __init__(self, net_name, device):
        super().__init__()
        self.net = get_net(net_name).to(device)
        self.log = pd.DataFrame()
        self.device = device

    def save(self, path):
        torch.save({'net': self.net.state_dict(),
                    'log': self.log}, path)
        print('Model saved to {}'.format(path))

    def load(self, path):
        if os.path.exists(path):
            file = torch.load(path)
            self.net.load_state_dict(file['net'])
            self.log = file['log']
            print('Model loaded from {}'.format(path))
            return True
        else:
            return False

    def train(self, dataloader, preprocess=None, args={}):
        default_args = {
            'num_batches': -1,
            'lr': 1e-2,
            'momentum': 0.9,
            'wd': 5e-4,
        }
        default_args.update(args); args = default_args
        num_batches = args['num_batches'] if 'num_batches' in args.keys() else -1
        optimizer = torch.optim.SGD(self.net.parameters(), lr=args['lr'],
                            momentum=args['momentum'], weight_decay=args['wd'])
        criterion = nn.CrossEntropyLoss()
        self.net.train()
        loss_total = 0.
        # train for one epoch with optional early stop
        for batch_id, (data, label) in enumerate(dataloader):
            if num_batches > 0 and batch_id >= num_batches:
                break
            if preprocess != None:
                for fn in preprocess:
                    data, label = fn(data, label)
            data, label = data.to(self.device), label.to(self.device)
            optimizer.zero_grad()
            output = self.net(data)
            loss = criterion(output, label)
            loss.backward()
            optimizer.step()
            loss_total += loss.item()
        loss_avg = loss_total / (batch_id + 1)
        return loss_avg
    
    def test(self, dataloader, preprocess=None, args={}):
        default_args = {
            'num_batches': -1,
        }
        default_args.update(args); args = default_args
        num_batches = args['num_batches'] if 'num_batches' in args.keys() else -1
        criterion = nn.CrossEntropyLoss()
        self.net.eval()
        loss_total = 0.
        accu_total = 0.
        # test for one epoch with optional early stop
        for batch_id, (data, label) in enumerate(dataloader):
            if num_batches > 0 and batch_id >= num_batches:
                break
            if preprocess != None:
                for fn in preprocess:
                    data, label = fn(data, label)
            data, label = data.to(self.device), label.to(self.device)
            output = self.net(data)
            loss = criterion(output, label)
            loss_total += loss.item()
            _, pred = torch.max(output, 1)
            accu = torch.mean((pred == label) * 1.)
            accu_total += accu.item()
        loss_avg = loss_total / (batch_id + 1)
        accu_avg = accu_total / (batch_id + 1)
        return loss_avg, accu_avg

    def train_loop(self, dataset, preprocess=None, args={}):
        default_args = {
            'num_epochs': 20
        }
        default_args.update(args); args = default_args
        for epoch in range(args['num_epochs']):
            loss = self.train(dataset.trainloader(), preprocess, args)
            test_loss, test_accu = self.test(dataset.testloader(), preprocess)
            log_item = {'train_loss': loss, 'test_loss': test_loss, 'test_accu': test_accu}
            print(log_item)
            self.log.append(log_item, ignore_index=True)
