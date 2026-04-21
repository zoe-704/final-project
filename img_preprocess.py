import cv2
import numpy as np
import matplotlib.pyplot as plt

def preprocess_image(image_path, target_size=(64, 64)):
    # 1. load image
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # 2. resize and
    img = cv2.resize(img, target_size)
    
    img = img.astype('float_32') / 255.0

    # 4. add channel dimension -> (1, H, W)
    img = np.expand_dims(img, axis=0)
    # 5. add batch dimension -> (1, 1, H, W)
    img = np.expand_dims(img, axis=0)

    return img
