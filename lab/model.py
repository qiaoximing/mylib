import torch
import torch.nn as nn
import torchvision
import pandas as pd

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
    return net

class Model():
    def __init__(self, net_name):
        super().__init__()
        self.net = get_net(net_name)
        self.log = pd.DataFrame()

    def save(self, path):
        torch.save({'net': self.net.state_dict(),
                    'log': self.log}, path)

    def load(self, path):
        tmp = torch.load(path)
        self.net.load_state_dict(tmp.net)
        self.log = tmp.log