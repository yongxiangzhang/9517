import numpy as np
import matplotlib.pyplot as plt
import os

X_test = np.load('base_model_unet/X_test.npy')
y_test = np.load('base_model_unet/y_test.npy')
y_pred = np.load('base_model_unet/y_pred.npy')

# 二值化预测
y_pred_bin = (y_pred > 0.5).astype(np.uint8)

# 可视化前5张
for i in range(min(5, X_test.shape[0])):
    img = X_test[i]
    mask = y_test[i]
    pred = y_pred_bin[i]
    plt.figure(figsize=(12,4))
    plt.subplot(1,3,1)
    plt.title('Image')
    plt.imshow(img[..., ::-1])  # BGR to RGB
    plt.axis('off')
    plt.subplot(1,3,2)
    plt.title('Ground Truth')
    plt.imshow(mask, cmap='gray')
    plt.axis('off')
    plt.subplot(1,3,3)
    plt.title('Prediction')
    plt.imshow(pred, cmap='gray')
    plt.axis('off')
    plt.tight_layout()
    plt.show() 