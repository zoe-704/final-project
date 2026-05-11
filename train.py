import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import pickle
import random
import argparse

CLASS_NAMES = ["rock", "paper", "scissors"]

# PREPROCESSING
# Convert a raw image file into expected tensor shape for CNN
# (1, 1, 64, 64) → (batch_size, channels, height, width)

def preprocess_image(image_path, target_size=(64, 64)):
    # load as grayscale image (one channel)
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # resize to 64x64
    img = cv2.resize(img, target_size)

    # normalize pixel values from [0, 255] to [0.0, 1.0]
    img = img.astype('float32') / 255.0

    # add channel dimension (64, 64) → (1, 64, 64) (channels, height, width)
    img = np.expand_dims(img, axis=0)

    # add batch dimension (1, 64, 64) → (1, 1, 64, 64) batch size is 1 for single images
    img = np.expand_dims(img, axis=0)
    return img

# FORWARD PASS
# Each function computes one layer's output given its input
# input → conv1 → relu → pool → conv2 → relu → pool →
# conv3 → relu → pool → conv4 → relu → pool →
# flatten → dense1 → relu → dense2 → relu →
# dense3 → relu → dense4 → softmax → loss

"""
slide each filter (kernel) across input feature map to detect features like edges or patterns
computes dot product between filter and local region of input for one value in output feature map
input: (batch, in_channels, height, width)
kernel: (out_channels, in_channels, kH, kW)
output: (batch, out_channels, out_height, out_width)
padding=1 and 3x3 kernel, so output H/W = input H/W
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

 # leaky relu passes 1% of negative values through to keep gradients alive across all layers
 def relu(x):
    return np.where(x > 0, x, 0.01 * x)  

# take max value in each block → reduces computation
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

# transform 4d tensor (N, C, H, W) → (N, features)
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
Backpropagation to compute how much each weight contributed to loss and adjust weights to reduce loss
Use chain rule for dout input and dx output
loss → softmax+ce → dense4 → relu → dense3 → relu → dense2 → relu → dense1 → flatten → 
pool4 → relu → conv4 → pool3 → relu → conv3 → pool2 → relu → conv2 → pool1 → relu → conv1

dout = “derivative of the loss with respect to the output of the current layer”
"""
# Combined gradient of softmax and cross-entropy loss
# -1 from predicted probability of correct class, divide by batch size
def softmax_cross_entropy_backward(pred, labels):
    grad = pred.copy() # predicted probabilities (N, C)
    grad[np.arange(len(labels)), labels] -= 1 # -1 from correct class
    grad /= len(labels)
    return grad # return dL/d(logits) gradient of loss for raw scores

"""
dout: gradient from layer ahead (dL/d(output))
x: the input that was used in the forward pass
W: the weight matrix used in the forward pass

dW = x.T @ dout — how much to adjust each weight (input * upstream gradient)
db = sum(dout) — how much to adjust each bias (sum upstream gradients across batch)
dx = dout @ W.T - gradient to pass back to the previous layer (how much each input contributed to the loss)
"""
def dense_backward(dout, x, W):
    dW = x.T @ dout # grad wrt weights W (how much to adjust W)
    db = np.sum(dout, axis=0, keepdims=True) # wrt bias (sum across batch)
    dx = dout @ W.T # wrt input (passed to prev layer)
    return dW, db, dx

# reshapes the gradient from 2D (batch, features) back to 4D (batch, channels, H, W)
def flatten_backward(dout, original_shape):
    return dout.reshape(original_shape)
    
# only pos with max in forward pass receives gradient (everything else gets 0) bc non-max values didn't affect the output
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

# gradient scaled by 0.01 to match 0.01 slope in the forward pass
def relu_backward(dout, x):
    dx = dout.copy()
    dx[x <= 0] *= 0.01 # pass 1% of gradient through dead neurons
    return dx

""" 
dk (kernel gradient): calculate how much each weight contributed to loss for each filter
dx (input gradient): determine which gradient should flow back to prev layer
"""
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


# TRAIN STEP
# a full forward and backward pass on a single image
# returns updated weights after one gradient descent step
def train_step(img, labels, K1, K2, K3, K4, W1, W2, W3, W4, b1, b2, b3, b4, lr=0.001):
    # FORWARD
    # conv block 1: (1,1,64,64) → conv(pad=1) → (1,8,32,32) → relu → pool → (1, 8, 32, 32)
    conv1_out = conv2D(img, K1, padding=1)
    relu1_out = relu(conv1_out)
    pool1_out = max_pooling(relu1_out, 2, 2)
 
    # conv block 2: (1,8,32,32) → conv(pad=1) → (1,16,32,32) → relu → pool → (1, 16, 16, 16)
    conv2_out = conv2D(pool1_out, K2, padding=1)
    relu2_out = relu(conv2_out)
    pool2_out = max_pooling(relu2_out, 2, 2)

    # conv block 3: (1,16,16,16) → conv(pad=1) → (1,32,16,16) → relu → pool → (1,32,8,8)
    conv3_out = conv2D(pool2_out, K3, padding=1)
    relu3_out = relu(conv3_out)
    pool3_out = max_pooling(relu3_out, 2, 2)

    # conv block 4: (1,32,8,8) → conv(pad=1) → (1,64,8,8) → relu → pool → (1,64,4,4)
    conv4_out = conv2D(pool3_out, K4, padding=1)
    relu4_out = relu(conv4_out)
    pool4_out = max_pooling(relu4_out, 2, 2)

    # flatten: (1,64,4,4) → (1,1024)
    pool4_shape = pool4_out.shape # save for unflattening
    flat = flatten(pool4_out)

    # dense block 1: (1,1024) → (1,256) → relu
    dense1_out = dense(flat, W1, b1) 
    drelu1_out = relu(dense1_out)

    # dense block 2: (1,256) → (1,128) → relu
    dense2_out = dense(drelu1_out, W2, b2) 
    drelu2_out = relu(dense2_out)

    # dense block 3: (1,128) → (1,64) → relu
    dense3_out = dense(drelu2_out, W3, b3)
    drelu3_out = relu(dense3_out)

    # output layer: (1,64) → (1,3) → softmax → probabilities
    logits = dense(drelu3_out, W4, b4)
    pred = softmax(logits)
    loss = cross_entropy(pred, labels)
 
    # BACKWARD
    # work backwards from loss, passing dout through each layer
    # each layer returns its weight gradients (dW, db) and input gradient (dx)

    # gradient of loss w.r.t. logits (combined softmax + cross-entropy)
    dlogits = softmax_cross_entropy_backward(pred, labels)
    # dense4 backward: get dW4, db4, and gradient to pass to relu3
    dW4, db4, ddrelu3 = dense_backward(dlogits, drelu3_out, W4)

    # relu backward for dense3's activation
    ddense3 = relu_backward(ddrelu3, dense3_out)
    # dense3 backward
    dW3, db3, ddrelu2 = dense_backward(ddense3, drelu2_out, W3)

    # dense2
    ddense2 = relu_backward(ddrelu2, dense2_out)
    dW2, db2, ddrelu1 = dense_backward(ddense2, drelu1_out, W2)

    # dense1 backward: gradient reaches the flattened vector
    ddense1 = relu_backward(ddrelu1, dense1_out)
    dW1, db1, dflat = dense_backward(ddense1, flat, W1)
    
    # unflatten: (1,1024) → (1,64,4,4) to flow back into conv4's pool
    dpool4 = flatten_backward(dflat, pool4_shape)

    # conv block 4 backward (pool → relu → conv)
    drelu4 = max_pooling_backward(dpool4, relu4_out, 2, 2)
    dconv4 = relu_backward(drelu4, conv4_out)
    dpool3, dK4 = conv2D_backward(dconv4, pool3_out, K4, padding=1)

    # conv block 3
    drelu3 = max_pooling_backward(dpool3, relu3_out, 2, 2)
    dconv3 = relu_backward(drelu3, conv3_out)
    dpool2, dK3 = conv2D_backward(dconv3, pool2_out, K3, padding=1)

    # conv block 2
    drelu2 = max_pooling_backward(dpool2, relu2_out, 2, 2)
    dconv2 = relu_backward(drelu2, conv2_out)
    dpool1, dK2 = conv2D_backward(dconv2, pool1_out, K2, padding=1)

    # conv block 1 backward (dx is gradient w.r.t. raw input not used)
    drelu1 = max_pooling_backward(dpool1, relu1_out, 2, 2)
    dconv1 = relu_backward(drelu1, conv1_out)
    _, dK1 = conv2D_backward(dconv1, img, K1, padding=1)

    # update
    # print(f" dK1:{np.abs(dK1).mean():.5f} dK4:{np.abs(dK4).mean():.5f} dW1:{np.abs(dW1).mean():.5f} dW4:{np.abs(dW4).mean():.5f}")

    # GRADIENT CLIPPING
    # cap gradients to [-5, 5] to prevent occasional large gradient spikes
    clip = 5.0
    dK1 = np.clip(dK1, -clip, clip);  dK2 = np.clip(dK2, -clip, clip)
    dK3 = np.clip(dK3, -clip, clip);  dK4 = np.clip(dK4, -clip, clip)
    dW1 = np.clip(dW1, -clip, clip);  dW2 = np.clip(dW2, -clip, clip)
    dW3 = np.clip(dW3, -clip, clip);  dW4 = np.clip(dW4, -clip, clip)
    db1 = np.clip(db1, -clip, clip);  db2 = np.clip(db2, -clip, clip)
    db3 = np.clip(db3, -clip, clip);  db4 = np.clip(db4, -clip, clip)

    # WEIGHT UPDATE (SGD) 
    # move each weight in the direction that reduces loss: weight -= lr * gradient
    K1 -= lr*dK1;  K2 -= lr*dK2;  K3 -= lr*dK3;  K4 -= lr*dK4
    W1 -= lr*dW1;  W2 -= lr*dW2;  W3 -= lr*dW3;  W4 -= lr*dW4
    b1 -= lr*db1;  b2 -= lr*db2;  b3 -= lr*db3;  b4 -= lr*db4
    return loss, K1, K2, K3, K4, W1, W2, W3, W4, b1, b2, b3, b4

# forward pass only
def predict(img, K1, K2, K3, K4, W1, W2, W3, W4, b1, b2, b3, b4):
    # 4 conv layers
    conv1_out = conv2D(img, K1, padding=1)
    pool1_out = max_pooling(relu(conv1_out), 2, 2)
 
    conv2_out = conv2D(pool1_out, K2, padding=1)
    pool2_out = max_pooling(relu(conv2_out), 2, 2)
 
    conv3_out = conv2D(pool2_out, K3, padding=1)
    pool3_out = max_pooling(relu(conv3_out), 2, 2)
    
    conv4_out = conv2D(pool3_out, K4, padding=1)
    pool4_out = max_pooling(relu(conv4_out), 2, 2)

    # 4 dense layers
    flat = flatten(pool4_out)
    out = relu(dense(flat, W1, b1))
    out = relu(dense(out, W2, b2))
    out = relu(dense(out, W3, b3))
    logits = dense(out, W4, b4)

    probs = softmax(logits)

    pred_class = np.argmax(probs, axis=1) # index of highest probabili
    return pred_class, probs

# RANDOM SAMPLE
# randomly samples N images per class from train/
# provides model with more variety over time
def load_dataset(folder, samples_per_class=300):
    images, labels = [], []
    for label_idx, class_name in enumerate(CLASS_NAMES):
        class_dir = os.path.join(folder, class_name)
        if not os.path.exists(class_dir):
            print(f"Warning: {class_dir} not found")
            continue
        # list all valid image files in class folder
        all_files = [f for f in os.listdir(class_dir)
                     if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        # randomly pick samples_per_class files (or all if fewer available)
        sampled = random.sample(all_files, min(samples_per_class, len(all_files)))

        for fname in sampled:
            images.append(preprocess_image(os.path.join(class_dir, fname)))
            labels.append(label_idx)

        print(f" {class_name}: sampled {len(sampled)}/{len(all_files)}")

    return images, np.array(labels)

# AUGMENTATION
# randomly transforms training images to make model more robust
# per-image during training so model doesn't see same image twice despite resampling same files
def augment(img):
    if np.random.rand() > 0.5:
        img = img[:, :, :, ::-1] # horizontal flip
    if np.random.rand() > 0.5:
        img = img[:, :, ::-1, :] # vertical flip
    img = img + np.random.uniform(-0.05, 0.05) # brightness
    return np.clip(img, 0, 1)

# TRAIN
# full training loop: for each epoch, sample fresh images, shuffle them, run one train_step per image, and print average loss
# model sees different images every epoch
def train(train_folder, epochs=30, lr=0.001, resume=False, samples_per_class=300):
    if resume and os.path.exists("model.pkl"):
        print("Resuming from model.pkl...")
        K1,K2,K3,K4, W1,W2,W3,W4, b1,b2,b3,b4 = load_model()
    else:
        print("Starting fresh...")

        # He initialization — scales weights to stop gradients vanishing in deep nets
        K1 = np.random.randn(8, 1, 3, 3) * np.sqrt(2.0 / (1*9))
        K2 = np.random.randn(16, 8, 3, 3) * np.sqrt(2.0 / (8*9))
        K3 = np.random.randn(32, 16, 3, 3) * np.sqrt(2.0 / (16*9))
        K4 = np.random.randn(64, 32, 3, 3) * np.sqrt(2.0 / (32*9))

        W1 = np.random.randn(1024, 256) * np.sqrt(2.0 / 1024)
        W2 = np.random.randn(256, 128) * np.sqrt(2.0 / 256)
        W3 = np.random.randn(128, 64) * np.sqrt(2.0 / 128)
        W4 = np.random.randn(64, 3) * np.sqrt(2.0 / 64)

        b1 = np.zeros((1, 256))
        b2 = np.zeros((1, 128))
        b3 = np.zeros((1, 64))
        b4 = np.zeros((1, 3))

    for epoch in range(epochs):
        images, labels = load_dataset(train_folder, samples_per_class)
        indices = np.random.permutation(len(images))  
        total_loss = 0.0
        for idx in indices:
            # aug = augment(images[idx]) # flip around for more variation
            loss, K1,K2,K3,K4, W1,W2,W3,W4, b1,b2,b3,b4 = train_step(
                images[idx], np.array([labels[idx]]), K1,K2,K3,K4, W1,W2,W3,W4, b1,b2,b3,b4, lr=lr
            )
            total_loss += loss
        print(f"epoch {epoch+1:02d}/{epochs} loss: {total_loss/len(images):.4f}")

    save_model(K1,K2,K3,K4, W1,W2,W3,W4, b1,b2,b3,b4)
    return K1,K2,K3,K4, W1,W2,W3,W4, b1,b2,b3,b4

# EVALUATE
# runs predict() on every test image and reports per-class accuracy
# test images were not used during training
def evaluate(test_folder, K1, K2, K3, K4, W1, W2, W3, W4, b1, b2, b3, b4):
    images, labels = load_dataset(test_folder, samples_per_class=9999) # load all
    correct = 0
    class_correct = [0, 0, 0]
    class_total = [0, 0, 0]

    for img, true_label in zip(images, labels):
        pred_class, _ = predict(img, K1,K2,K3,K4, W1,W2,W3,W4, b1,b2,b3,b4)
        predicted = pred_class[0]
        class_total[true_label] += 1
        if pred_class[0] == true_label:
            correct += 1
            class_correct[true_label] += 1

    print(f"\nOverall accuracy: {correct}/{len(images)} ({correct/len(images)*100:.1f}%)")
    for i, name in enumerate(CLASS_NAMES):
        if class_total[i] > 0:
            print(f" {name:10s}: {class_correct[i]}/{class_total[i]} ({class_correct[i]/class_total[i]*100:.1f}%)")

# SAVE/LOAD MODEL
# stores all learned weights using pickle
# load_model() lets training resume from where it left off
def save_model(K1, K2, K3, K4, W1, W2, W3, W4, b1, b2, b3, b4, path="model.pkl"):
    with open(path, "wb") as f:
        pickle.dump({"K1":K1,"K2":K2,"K3":K3,"K4":K4,
                     "W1":W1,"W2":W2,"W3":W3,"W4":W4,
                     "b1":b1,"b2":b2,"b3":b3,"b4":b4}, f)
    print(f"Model saved to {path}")

def load_model(path="model.pkl"):
    with open(path, "rb") as f:
        d = pickle.load(f)
    return d["K1"],d["K2"],d["K3"],d["K4"],d["W1"],d["W2"],d["W3"],d["W4"],d["b1"],d["b2"],d["b3"],d["b4"]

# RUN
# python forward.py
# python forward.py --epochs 30 --lr 0.0005 --resume --samples_per_class 500
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train RPS classifier")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--resume", action="store_true")  # flag, no value needed
    parser.add_argument("--samples_per_class", type=int, default=300)
    args = parser.parse_args()

    K1,K2,K3,K4, W1,W2,W3,W4, b1,b2,b3,b4 = train(
        "train", epochs=args.epochs, lr=args.lr, resume=args.resume, samples_per_class=args.samples_per_class
    )
    evaluate("test", K1,K2,K3,K4, W1,W2,W3,W4, b1,b2,b3,b4)