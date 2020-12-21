import torch
import torch.nn as nn
import torchvision.models as models

def Model(pretrain):
    classifier = models.resnet34(pretrained=pretrain)
    num_ftrs = classifier.fc.in_features
    classifier.fc = nn.Linear(num_ftrs,5)
    return classifier