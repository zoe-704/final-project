# img_preprocess
import cv2
import numpy as np
import matplotlib.pyplot as plt

def preprocess_image(image_path, target_size=(64, 64)):
    # 1. load image
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # 2. resize and normalize
    img = cv2.resize(img, target_size)
    img = img.astype('float32') / 255.0

    # 4. add channel dimension -> (1, H, W)
    img = np.expand_dims(img, axis=0)
    # 5. add batch dimension -> (1, 1, H, W)
    img = np.expand_dims(img, axis=0)
    return img

# FORWARD
"""
slide filter (kernel) across  image to detect features like edges or patterns and produce a feature map
output is 4d tensor (batch_size, out_channels, out_height, out_width)
input (1, 1, 64, 64) & kernel (8, 1, 3, 3) --> output (1, 8, 62, 62)
"""
def conv2D(input, kernel, stride=1, padding=0): 
    batch_size, in_depth, in_height, in_width = input.shape
    out_channels, __, kernel_height, kernel_width = kernel.shape

    # pad input before applying convolution filter without changing batch size and channel dimensions
    input_padded = np.pad(input, ((0, 0), (0, 0), (padding, padding), (padding, padding)), mode='constant')

    out_height = (in_height + 2*padding - kernel_height) // stride + 1 
    out_width = (in_width + 2*padding - kernel_width) // stride + 1 
    output = np.zeros((batch_size, out_channels, out_height, out_width))
    
    # iterate through h and w of output feature map for each image
    for n in range(batch_size): 
        for c in range(out_channels):
            for i in range(out_height):
                for j in range(out_width):
                    # dot product multiplies element in region with element in the kernel and sums all values
                    region = input_padded[n, :, i * stride:i * stride + kernel_height, j * stride:j * stride + kernel_width]
                    output[n, c, i, j] = np.sum(region * kernel[c])
    return output

# make negative values 0 and introduce non-linearity
def relu(x):
    return np.maximum(0, x)

# take max value in each block --> reduces computation
def max_pooling(input, pool_size, stride):
    batch_size, in_depth, in_height, in_width = input.shape
    out_height = (in_height - pool_size) // stride + 1
    out_width = (in_width - pool_size) // stride + 1
    output = np.zeros((batch_size, in_depth, out_height, out_width))
    # iterate through h and w of feature map for each image
    for n in range(batch_size): 
        for c in range(in_depth):
            for i in range(out_height):
                for j in range(out_width):
                        region = input[n, :, i * stride:i * stride + pool_size, j * stride:j * stride + pool_size]
                        # find the max value of each region (most significant feature of area)
                        output[n, c, i, j] = np.max(region)
    return output

# transform 4d tensor (N, C, H, W) --> (N, features)
def flatten(x):
    return x.reshape(x.shape[0], -1)

# weighted sum of all input features and bias
def dense(x, W, b):
    return x @ W + b

# equation: softmax(x_i) = e^(x_i) / sum of [e^(x_j)]
def softmax(x):
    exp = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exp / np.sum(exp, axis=1, keepdims=True)

# equation: L = -(1/N) * sum of [log(probs_i)]
def cross_entropy(pred, labels):
    N = pred.shape[0]
    probs = pred[np.arange(N), labels]
    loss = -np.mean(np.log(probs + 1e-8))
    return loss

# BACKWARD
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


def predict(img, K, W, b):
    # forward pass
    conv_out = conv2D(img, K)
    relu_out = relu(conv_out)
    pool_out = max_pooling(relu_out, 2, 2) # max value in each 2x2 block

    flat = flatten(pool_out)
    logits = dense(flat, W, b)
    probs = softmax(logits)

    pred_class = np.argmax(probs, axis=1)
    
    return pred_class, probs


# test
img = preprocess_image("test.jpg")

# conv kernel
# 8 filters (8 dif features), 1 input channel (grayscale), 3x3 filter
K = np.random.randn(8, 1, 3, 3) * 0. 

# after conv+pool --> (8, 31, 31)
# 64×64 --> (3x3 conv) 62×62 --> (2×2 max pool) 31×31
flatten_size = 8 * 31 * 31
 
# dense
W = np.random.randn(flatten_size, 10) * 0.1 # 7688 inputs → 10 output classes
b = np.zeros((1, 3)) # one bias per 3 class (rock, paper, scissors)

pred, probs = predict(img, K, W, b)

print("Predicted class:", pred[0])
print("Probabilities:", probs)

plt.imshow(img[0, 0], cmap='gray')
plt.title(f"Prediction: {pred[0]}")
plt.axis('off')
plt.show()