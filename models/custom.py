import torch
import torch.nn as nn
import torch.nn.functional as F

# custom model
class Model(nn.Module):
    def __init__(self, num_classes):
        super(Model, self).__init__()
        self.num_classes = num_classes
        
        # input block
        self.convblock1 = nn.Sequential(
                          nn.Conv2d(in_channels=3, out_channels=32, kernel_size=(3,3), padding=1, bias=False),
                          nn.SELU(),
                          nn.BatchNorm2d(32)        
        )
        self.convblock2 = nn.Sequential(
                          nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3,3), padding=1, bias=False),
                          nn.SELU(),
                          nn.BatchNorm2d(64)        
        )
        
        # transition block1
        self.pool1 = nn.MaxPool2d(2,2) # output_size = 24 RF=7
        self.convblock3 = nn.Sequential(
                          nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3,3), padding=1, bias=False),
                          nn.SELU(),
                          nn.BatchNorm2d(128)
        )
        self.convblock4 = nn.Sequential(
                          nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(3,3), padding=1, bias=False),
                          nn.SELU(),
                          nn.BatchNorm2d(256)
        )
        self.convblock5 = nn.Sequential(
                          nn.Conv2d(in_channels=256, out_channels=512, kernel_size=(1,1), padding=1, bias=False),
                          nn.SELU(),
                          nn.BatchNorm2d(512)
        )
        
        # transition block2
        self.pool2 = nn.MaxPool2d(2,2) # output_size = 12 RF=20
        
        # convolution block
        self.convblock6 = nn.Sequential(
                          nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=(3,3), padding=1, bias=False),
                          nn.SELU(),
                          nn.BatchNorm2d(1024)
        )
        self.convblock7 = nn.Sequential(
                          nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=(3,3), padding=1, bias=False),
                          nn.SELU(),
                          nn.BatchNorm2d(1024)
        )
        
        # transition block3
        self.pool3 = nn.MaxPool2d(2,2) # output_size = 6 RF=32
        self.convblock8 = nn.Sequential(
                          nn.Conv2d(in_channels=1024, out_channels=512, kernel_size=(3,3), padding=1, bias=False),
                          nn.SELU(),
                          nn.BatchNorm2d(512)
        )
        self.convblock9 = nn.Sequential(
                          nn.Conv2d(in_channels=512, out_channels=256, kernel_size=(3,3), padding=0, bias=False),
                          nn.SELU(),
                          nn.BatchNorm2d(256)
        )
        
        self.gap = nn.Sequential(
                   nn.AvgPool2d(kernel_size=5)
        )
        self.convblock10 = nn.Sequential(
                           nn.Conv2d(in_channels=256, out_channels=num_classes, kernel_size=(1,1), padding=0, bias=False)
        )
        
    def forward(self, x):
        x = self.convblock1(x)
        x = self.convblock2(x)
        x = self.pool1(x)
        x = self.convblock3(x)
        x = self.convblock4(x)
        x = self.convblock5(x)
        x = self.pool2(x)
        x = self.convblock6(x)
        x = self.convblock7(x)
        x = self.pool3(x)
        x = self.convblock8(x)
        x = self.convblock9(x)
        x = self.gap(x)
        x = self.convblock10(x) # shape = torch.size([64, num_classes, 1, 1])
        x = x.view(-1, self.num_classes) # shape = torch.size([64, num_classes])
        return F.log_softmax(x, dim=-1)