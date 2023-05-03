import torch
import torch.nn.init as init
import math

class Conv2dZ2P4(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, g_type="p4",  
                 dilation=1, groups=1, bias=False, device="cuda", dtype=None, *args, **kwargs):
        super().__init__()
        assert g_type == "p4" or g_type == "p4m"
        
        # define the layer weight
        w = torch.empty(out_channels, in_channels, kernel_size, kernel_size)
        self.weight = torch.nn.Parameter(w).to(device)
        
        self.g_type = g_type 
        self.get_kernel = get_p8weight if self.g_type == "p4m" else get_p4weight
        self.gconv_dim = 8 if self.g_type == "p4m" else 4
        self.__args = args
        self.__kwargs = kwargs
        
        if bias:
            self.bias = torch.nn.Parameter(torch.zeros(out_channels))
        else:
            self.bias = None
            
        self.__weight_initialization()
    
    def forward(self, x):
        w = self.get_kernel(self.weight)
        
        padding_dim = w.shape[-1] // 2
        y = torch.nn.functional.conv2d(x, w, *self.__args, **self.__kwargs)
        y = y.view(y.size(0), -1, 4, y.size(2), y.size(3))
        if self.bias is not None:
            y = y + self.bias.view(1, -1, 1, 1, 1)
        return y
            
    def __weight_initialization(self):
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None: 
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)
            


class Conv2dP4P4(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, g_type="p4", bias=False, device="cuda", *args, **kwargs):
        assert g_type == "p4" or g_type == "p4m"
        super().__init__()
        self.out_channels = out_channels
        w = torch.empty(out_channels*4, in_channels, kernel_size, kernel_size)
        self.weight = torch.nn.Parameter(w).to(device)
        
        self.g_type = g_type 
        self.get_kernel = get_p8weight if self.g_type == "p4m" else get_p4weight
        self.gconv_dim = 8 if self.g_type == "p4m" else 4
        self.__args = args
        self.__kwargs = kwargs
        
        if bias:
            self.bias = torch.nn.Parameter(torch.zeros(out_channels))
        else:
            self.bias = None
            
        self.__weight_initialization()
    
    def forward(self, x):
        b = self.bias.repeat(self.gconv_dim) if self.bias is not None else None
        w = self.weight
        B, C, _, H, W = x.shape
        y = None
        
        device = x.device
        
        padding_dim = w.shape[-1] // 2
        
        for i in range(4):
            _, _, _, H, W = x.shape
            x_ = g_rot4(x, -i)

            x_ = x_.transpose(1,2).reshape(B, C * self.gconv_dim, H, W)

            t = torch.nn.functional.conv2d(x_, w, groups=self.gconv_dim, *self.__args, **self.__kwargs)
            _, _, H, W = t.shape
            t = t.reshape(B, -1, 4, H, W).sum(dim=2)

            if y is None: 
                y = torch.zeros(B, self.out_channels, 4, H, W).to(device)
            y[:, :, i, :, :] = t
            
        if self.bias is not None:
            y = y + b
        return y
            
    def __weight_initialization(self):
        
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        
        if self.bias is not None: 
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)    
          
            
class MaxPoolingP4(torch.nn.Module):
    def __init__(self, kernel_size=(2,2), stride=2):
        super().__init__()
        self.pool = torch.nn.MaxPool2d(kernel_size)
        
    def forward(self, x):
        B, C, _, H, W = x.shape
        x = x.view(B, -1, H, W)  
        x_pool = self.pool(x)
        _, _, H, W = x_pool.shape
        return x_pool.view(B, C, 4, H, W)
    
class AvgPoolingP4(torch.nn.Module):
    def __init__(self, kernel_size=(2,2), stride=2):
        super().__init__()
        self.pool = torch.nn.AvgPool2d(kernel_size)
        
    def forward(self, x):
        B, C, _, H, W = x.shape
        x = x.view(B, -1, H, W)  
        x_pool = self.pool(x)
        _, _, H, W = x_pool.shape
        return x_pool.view(B, C, 4, H, W)
            
            
def get_p4weight(w):
    # input: [C, K, H, W]
    # output: [4*C, K, H, W]
    
    # rotate the input weight
    ws = [torch.rot90(w, k, (-2, -1)) for k in range(4)]
    return torch.cat(ws, 1).view(-1, w.size(1), w.size(2), w.size(3))

def get_p8weight(w):
    # input: [K, C, H, W]
    # output: [8*K, C, H, W]
    w_p4 = get_p4weight(w)
    return torch.cat([w_p4, torch.flip(w_p4, dims=(-1,))], 1).view(-1, w.size(1), w.size(2), w.size(3))

def g_rot4(x, k, reverse=False):
    device = x.device
    if reverse: 
        k = -k
    x = torch.rot90(x, k, (-2, -1))
    return torch.roll(x, k, dims=-3).to(device)

