import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
from sklearn.model_selection import train_test_split

# 路径
IMAGE_DIR = 'USA_segmentation/NRG_images'
MASK_DIR = 'USA_segmentation/masks'
PRED_NPZ = 'base_model_xgb/xgb_pred.npz'

def compute_iou(y_true, y_pred):
    intersection = np.logical_and(y_true, y_pred).sum()
    union = np.logical_or(y_true, y_pred).sum()
    if union == 0:
        return 1.0 if intersection == 0 else 0.0
    return intersection / union

def get_test_split():
    image_files = sorted([os.path.join(IMAGE_DIR, f) for f in os.listdir(IMAGE_DIR) if f.endswith('.tif') or f.endswith('.tiff') or f.endswith('.png')])
    mask_files = sorted([os.path.join(MASK_DIR, f) for f in os.listdir(MASK_DIR) if f.endswith('.png') or f.endswith('.tif') or f.endswith('.tiff')])
    _, test_imgs, _, test_masks = train_test_split(image_files, mask_files, train_size=0.8, random_state=42)
    return test_imgs, test_masks

def main():
    # 加载预测结果
    data = np.load(PRED_NPZ)
    y_pred = data['y_pred']
    y_test = data['y_test']

    # 计算IoU
    iou = compute_iou(y_test, y_pred)
    print(f'IoU (Jaccard) score: {iou:.4f}')

    # 获取测试集图片和mask文件名
    test_imgs, test_masks = get_test_split()

    # 假设所有图片尺寸一致
    img0 = cv2.imread(test_imgs[0], cv2.IMREAD_UNCHANGED)
    h, w = img0.shape[:2]
    pixels_per_img = h * w

    # 可视化前5张
    for i in range(5):
        img = cv2.imread(test_imgs[i], cv2.IMREAD_UNCHANGED)
        mask = cv2.imread(test_masks[i], cv2.IMREAD_GRAYSCALE)
        pred_mask = y_pred[i * pixels_per_img : (i+1) * pixels_per_img].reshape(h, w)

        plt.figure(figsize=(12,4))
        plt.subplot(1,3,1)
        plt.title('Image')
        if img.shape[-1] == 3:
            plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        else:
            plt.imshow(img[..., :3])
        plt.axis('off')
        plt.subplot(1,3,2)
        plt.title('Ground Truth')
        plt.imshow(mask, cmap='gray')
        plt.axis('off')
        plt.subplot(1,3,3)
        plt.title('Prediction')
        plt.imshow(pred_mask, cmap='gray')
        plt.axis('off')
        plt.tight_layout()
        plt.show()

if __name__ == '__main__':
    main()