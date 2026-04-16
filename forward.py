import numpy as np

class conv2d:
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=8): 
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding


    def forward(self, x):
        # x shape C H W
        # returns out_channels H W
        self.x = x
        C, H, W = x.shape 
        K = self.kernel_size
        S = self.stride
        P = self.padding
        if P > 0:
            x_padded = np.pad(x, ((0, 0), (P, )))