import numpy as np

"""
    loss
    - softmax
    - dense
    - flatten
    - pooling
    - relu
    - conv
    
    dout = “derivative of the loss with respect to the output of the current layer”
    
    loss
    - dout (from next layer)
    - current layer
    - previous layer

"""

def softmax_cross_entropy_backward(pred, labels):
    grad = pred.copy() # predicted probabilities (N, C)
    grad[np.arange(len(labels)), labels] -= 1 # -1 from correct class
    grad /= len(labels)
    return grad # return dL/d(logits) gradient of loss for raw scores

def dense_backward(dout, x, W):
    dW = x.T @ dout # grad wrt weights W (how much to adjust W)
    db = np.sum(dout, axis=0, keepdims=True) # wrt bias (sum across batch)
    dx = dout @ W.T # wrt input (passed to prev layer)
    return dW, db, dx

def flatten_backward(dout, original_shape):
    return dout.reshape(original_shape)
    
def max_pooling_backward(dout, input, pool_size, stride):
    batch_size, depth, height, width = input.shape
    dx = np.zeros_like(input) # grad array like pooling input

    out_height = dout.shape[2]
    out_width = dout.shape[3]

    for n in range(batch_size):
        for c in range(depth):
            for i in range(out_height):
                for j in range(out_width):
                    # same region pooled in forward pass
                    region = input[n, c, i*stride:i*stride+pool_size, j*stride:j*stride+pool_size]
                    max_val = np.max(region)

                    for h in range(pool_size):
                        for w in range(pool_size):
                            if region[h, w] == max_val:
                                # max value gets gradient
                                # other values get 0 bc non-max values didn't affect output
                                dx[n, c, i*stride+h, j*stride+w] += dout[n, c, i, j]
    return dx

def relu_backward(dout, x):
    dx = dout.copy()
    dx[x <= 0] = 0 # zero neg grad 
    return dx
    
def conv2D_backward(dout, input, kernel, stride=1, padding=0):
    batch_size, in_depth, in_height, in_width = input.shape
    out_channels, _, kH, kW = kernel.shape
    # same padding as forward pass
    input_padded = np.pad(input, ((0,0),(0,0),(padding,padding),(padding,padding)))

    dx_padded = np.zeros_like(input_padded) # grad wrt input
    dk = np.zeros_like(kernel) # wrt kernel weights

    out_height = dout.shape[2]
    out_width = dout.shape[3]

    for n in range(batch_size):
        for c in range(out_channels):
            for i in range(out_height):
                for j in range(out_width):
                    # same region from forward pass
                    region = input_padded[n, :, i*stride:i*stride+kH, j*stride:j*stride+kW]
                    # how much to adjust kernel (input*grad)
                    dk[c] += region * dout[n, c, i, j]
                    # pass grad back to input (kernel*grad)
                    dx_padded[n, :, i*stride:i*stride+kH, j*stride:j*stride+kW] += kernel[c] * dout[n, c, i, j]

    # remove padding
    if padding > 0:
        dx = dx_padded[:, :, padding:-padding, padding:-padding] # strip padding
    else:
        dx = dx_padded
    # dx goes further back
    # dk updates kernel
    return dx, dk

# run backward pass
