import os
import numpy as np
import cv2
from sklearn.model_selection import train_test_split

# 数据路径
IMAGE_DIR = 'USA_segmentation/NRG_images'
MASK_DIR = 'USA_segmentation/masks'
SAVE_DIR = 'base_model_unet'
IMG_SIZE = 256  # 统一resize到256x256
TRAIN_SPLIT = 0.8

os.makedirs(SAVE_DIR, exist_ok=True)

def read_image(path):
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is not None and img.shape[-1] == 3:
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        return img.astype(np.float32) / 255.0
    else:
        raise ValueError(f"{path} 不是3通道图像！")

def read_mask(path):
    mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    mask = cv2.resize(mask, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_NEAREST)
    return (mask > 127).astype(np.uint8)

def main():
    image_files = sorted([os.path.join(IMAGE_DIR, f) for f in os.listdir(IMAGE_DIR) if f.endswith('.tif') or f.endswith('.tiff') or f.endswith('.png')])
    mask_files = sorted([os.path.join(MASK_DIR, f) for f in os.listdir(MASK_DIR) if f.endswith('.png') or f.endswith('.tif') or f.endswith('.tiff')])
    assert len(image_files) == len(mask_files), '图像和mask数量不一致'
    train_imgs, test_imgs, train_masks, test_masks = train_test_split(
        image_files, mask_files, train_size=TRAIN_SPLIT, random_state=42)
    # 训练集
    X_train = np.stack([read_image(p) for p in train_imgs])
    y_train = np.stack([read_mask(p) for p in train_masks])
    # 测试集
    X_test = np.stack([read_image(p) for p in test_imgs])
    y_test = np.stack([read_mask(p) for p in test_masks])
    # 保存
    np.save(os.path.join(SAVE_DIR, 'X_train.npy'), X_train)
    np.save(os.path.join(SAVE_DIR, 'y_train.npy'), y_train)
    np.save(os.path.join(SAVE_DIR, 'X_test.npy'), X_test)
    np.save(os.path.join(SAVE_DIR, 'y_test.npy'), y_test)
    print(f'训练集: {X_train.shape}, {y_train.shape} | 测试集: {X_test.shape}, {y_test.shape}')
    print(f'已保存到 {SAVE_DIR}/')

if __name__ == '__main__':
    main() 