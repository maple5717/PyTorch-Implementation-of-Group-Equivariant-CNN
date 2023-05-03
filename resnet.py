from gcnn.layers import *
import torch
import torch.nn.init as init
import math
from functools import reduce

class ResBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, use_gcnn, *args, **kwargs):
        super().__init__()
        if use_gcnn:
            self.conv1 = Conv2dP4P4(in_channels, out_channels, kernel_size, "p4", *args, **kwargs)
            self.conv2 = Conv2dP4P4(out_channels, out_channels, kernel_size, "p4", *args, **kwargs)
            self.bn1 = torch.nn.BatchNorm3d(out_channels)
            self.bn2 = torch.nn.BatchNorm3d(out_channels)
        else:
            self.conv1 = torch.nn.Conv2d(in_channels, out_channels, kernel_size, *args, **kwargs)
            self.conv2 = torch.nn.Conv2d(out_channels, out_channels, kernel_size, *args, **kwargs)
            self.bn1 = torch.nn.BatchNorm2d(out_channels)
            self.bn2 = torch.nn.BatchNorm2d(out_channels)
        
        self.relu = torch.nn.ReLU()
        
    def forward(self, x):
        y = (self.conv1(x))
        y = self.bn1(y)
        y = self.relu(y)
        y = self.bn2(self.conv2(y))
        
        return self.relu(x + y)
        
        
class ResNet(torch.nn.Module):
    def __init__(self, structure, use_gcnn, init_layer=(64, 7), input_dim=(3, 32, 32), device="cuda"):
        super().__init__()
        self.device = device
        self.init_layer = init_layer
        self.use_gcnn = use_gcnn
        self.create_layers(structure)
        
        sample_data = torch.randn(1, *input_dim).to(device)
        # print(self.conv_layers.device)
        output_shape = self.conv_layers(sample_data).cpu().shape
        fc_input_dim = (lambda tup: reduce(lambda x, y: x * y, tup))(output_shape)
        
        self.fc_layers = torch.nn.Sequential(
                torch.nn.Linear(fc_input_dim, out_features=1000), 
                torch.nn.Linear(1000, out_features=10)
        ).to(self.device)
        self.softmax = torch.nn.Softmax(dim=1)
    
        
    def create_layers(self, structure):
        init_conv_layer = Conv2dZ2P4 if self.use_gcnn else torch.nn.Conv2d
        conv_layer = Conv2dP4P4 if self.use_gcnn else torch.nn.Conv2d
        max_pooling_layer = MaxPoolingP4 if self.use_gcnn else torch.nn.MaxPool2d
        avg_pooling_layer = AvgPoolingP4 if self.use_gcnn else torch.nn.AvgPool2d
        layers_lst = []
        
        layers_lst.append(init_conv_layer(3, self.init_layer[0], self.init_layer[1], stride=2, padding=3))
        layers_lst.append(max_pooling_layer((3,3), 2))
        
        in_dim = self.init_layer[0]
        for x in structure:
            out_dim, stride  = x
            # print(out_dim)
            layers_lst += [conv_layer(in_dim, out_dim, kernel_size=3, stride=stride, padding=1)]  
            in_dim = out_dim
        
        # layers_lst.append(avg_pooling_layer((2, 2)))
        self.conv_layers = torch.nn.Sequential(*layers_lst).to(self.device)
        
    def forward(self, x):
        B = x.shape[0]
        x = self.conv_layers(x).reshape(B, -1)
        return (self.fc_layers(x))