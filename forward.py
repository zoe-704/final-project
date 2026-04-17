import numpy as np

    def conv2D(input, kernel, stride=1, padding=0): 
        batch_size, in_depth, in_height, in_width = input.shape
        out_channels, kernel_height, kernel_width = kernel.shape[0], kernel.shape[2], kernel.shape[3]

        # pad input before applying convolution filter without changing batch size and channel dimensions
        input_padded = np.pad(input, ((0, 0), (0, 0), (padding, padding), (padding, padding)), mode='constant')

        out_height = (in_height + 2*padding - kernel_height) // stride + 1 # // stride + 1??
        out_width = (in_width + 2*padding - kernel_width) // stride + 1 
        output = np.zeros((batch_size, out_channels, out_height, out_width))
        
        # iterate through h and w of output feature map for each image
        for n in range(batch_size): 
            for i in range(out_height):
                for j in range(out_width):
                    # kernel applied for each filter for each position in input
                    for c in range(out_channels):
                        # dot product multiplies element in region with element in the kernel and sums all values
                        region = input_padded[n, :, i * stride:i * stride + kernel_height, j * stride:j * stride + kernel_width]
                        output[n, c, i, j] = np.sum(region * kernel[c])
        return output

    def relu(x):
        return np.maximum(0, x)
    
    def max_pooling(input, pool_size, stride):
        batch_size, in_depth, in_height, in_width = input.shape
        out_height = in_height - pool_size // stride + 1
        out_width = in_width - pool_size // stride + 1
        output = np.zeros(batch_size, in_depth, out_height, out_width)
        # iterate through h and w of feature map for each image
        for n in range(batch_size): 
            for i in range(out_height):
                for j in range(out_width):
                        region = input_padded[n, :, i * stride:i * stride + pool_size, j * stride:j * stride + pool_size]
                        # find the max value of each region (most significant feature of area)
                        output[n, c, i, j] = np.max(region, axis=(1, 2))
        return output