import cv2
import numpy as np
import matplotlib.pyplot as plt

def preprocess_image(image_path, target_size=(64, 64)):
    # 1. load image
    img = cv2.imread(image_path)
    # 2. resize
    img = cv2.resize(img, target_size)
    # 3. normalize
    img = img.astype('float_32') / 255.0
    # 4. reshape ????
    img = np.expand_dims(img, axis=0)
    
    return img
