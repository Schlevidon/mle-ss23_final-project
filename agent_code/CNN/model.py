import torch
import torch.nn as nn
import torch.optim as optim
import os
import numpy as np

class CNN(nn.Module):

    def __init__(self, n_features, n_kernels_conv, n_kernels_pool, ROWS, COLS, activation=nn.ReLU):
        super().__init__()
        layers =[]
        pool_used = []
        n_layers = len(n_features)
        for i, (ch_in, ch_out, kernel_size_conv, kernel_size_pool) in enumerate(zip(n_features[:-2], n_features[1:-1],\
                                                                                     n_kernels_conv,n_kernels_pool)):
            
            conv = nn.Conv2d(ch_in, ch_out, kernel_size=kernel_size_conv, bias=True, stride=1, padding=0) #padding(?)
            layers.append(conv)
            if i != n_layers-2:
                layers.append(activation()) 
                layers.append(nn.MaxPool2d(kernel_size=kernel_size_pool))
                pool_used.append(kernel_size_pool)
                layers.append(activation()) # possibly not effective # max drop it later

        layers.append(nn.MaxPool2d(kernel_size=n_kernels_pool[-1]))
        pool_used.append(n_kernels_pool[-1])
        pool_used = np.array(pool_used)*2
        layers.append(nn.Flatten())
        layers.append(nn.Linear(int(ROWS*COLS/np.prod(pool_used)), n_features[-1], bias=True)) # change the image size dynamically
        #layers.append(nn.Sigmoid())
        self.layers = nn.ModuleList(layers)
        self.apply(self._init_weights)
        
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x.squeeze()
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=1.0)
            if module.bias is not None:
                module.bias.data.zero_()

        if isinstance(module, nn.Conv2d):
            module.weight.data.normal_(mean=0.0, std=1.0)
            if module.bias is not None:
                module.bias.data.zero_()

    def save(self, folder_path='./model', file_name='my-model.pt'):
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        file_name = os.path.join(folder_path, file_name)
        torch.save(self.state_dict(), file_name)
