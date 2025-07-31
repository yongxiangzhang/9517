import os
import numpy as np
import tifffile
import cv2
from sklearn.model_selection import train_test_split
import xgboost as xgb
import joblib
import time

# 数据路径
IMAGE_DIR = 'USA_segmentation/NRG_images'
MASK_DIR = 'USA_segmentation/masks'
TRAIN_SPLIT = 0.8

# 读取单张图像（3通道）
def read_image(path):
    ext = os.path.splitext(path)[-1].lower()
    if ext in ['.tif', '.tiff']:
        img = tifffile.imread(path)
    elif ext in ['.png', '.jpg', '.jpeg']:
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if img is not None and img.shape[-1] == 3:
            pass
        else:
            raise ValueError(f"{path} 不是3通道图像，请检查数据！")
    else:
        raise ValueError(f"不支持的文件格式: {path}")
    return img.astype(np.float32) / 255.0

# 读取单张mask
def read_mask(path):
    mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    return (mask > 127).astype(np.uint8)

# 提取增强特征和标签
def extract_features_and_labels(image_paths, mask_paths):
    features = []
    labels = []
    for img_path, mask_path in zip(image_paths, mask_paths):
        img = read_image(img_path)
        mask = read_mask(mask_path)
        h, w, c = img.shape
        if c != 3:
            raise ValueError(f"{img_path} 不是3通道图像！")
        # 1. RGB
        rgb = img.reshape(-1, 3)
        # 2. HSV
        hsv = cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_BGR2HSV).astype(np.float32) / 255.0
        hsv = hsv.reshape(-1, 3)
        # 3. Lab
        lab = cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_BGR2Lab).astype(np.float32) / 255.0
        lab = lab.reshape(-1, 3)
        # 4. Sobel边缘
        gray = cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
        sobelx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        sobel = np.sqrt(sobelx ** 2 + sobely ** 2).reshape(-1, 1)
        # 5. 局部均值/方差
        mean = cv2.blur(gray, (5, 5)).reshape(-1, 1)
        var = cv2.blur((gray - mean.reshape(h, w)) ** 2, (5, 5)).reshape(-1, 1)
        # 6. 空间坐标特征
        yy, xx = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')
        xx = (xx / w).reshape(-1, 1)
        yy = (yy / h).reshape(-1, 1)
        # 拼接所有特征
        feat = np.concatenate([rgb, hsv, lab, sobel, mean, var, xx, yy], axis=1)
        label = mask.flatten()
        features.append(feat)
        labels.append(label)
    features = np.concatenate(features, axis=0)
    labels = np.concatenate(labels, axis=0)
    return features, labels

def train_xgb(X, y, scale_pos_weight):
    print('训练XGBoost (GPU)...')
    t0 = time.time()
    dtrain = xgb.DMatrix(X, label=y)
    params = {
        'tree_method': 'hist',
        'device': 'cuda',
        'objective': 'binary:logistic',
        'eval_metric': 'logloss',
        'max_depth': 10,
        'learning_rate': 0.08,
        'lambda': 3,
        'alpha': 2,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'scale_pos_weight': scale_pos_weight,
        'verbosity': 1
    }
    num_boost_round = 250
    bst = xgb.train(params, dtrain, num_boost_round=num_boost_round)
    t1 = time.time()
    print(f'训练完成，用时{t1-t0:.2f}秒')
    return bst, t1-t0

def predict_xgb(bst, X):
    print('预测测试集...')
    t0 = time.time()
    dtest = xgb.DMatrix(X)
    y_pred = bst.predict(dtest)
    y_pred_label = (y_pred > 0.5).astype(np.uint8)
    t1 = time.time()
    print(f'预测完成，用时{t1-t0:.2f}秒')
    return y_pred_label, t1-t0

def main():
    image_files = sorted([os.path.join(IMAGE_DIR, f) for f in os.listdir(IMAGE_DIR) if f.endswith('.tif') or f.endswith('.tiff') or f.endswith('.png')])
    mask_files = sorted([os.path.join(MASK_DIR, f) for f in os.listdir(MASK_DIR) if f.endswith('.png') or f.endswith('.tif') or f.endswith('.tiff')])
    assert len(image_files) == len(mask_files), '图像和mask数量不一致'
    # 按图片划分训练/测试
    train_imgs, test_imgs, train_masks, test_masks = train_test_split(
        image_files, mask_files, train_size=TRAIN_SPLIT, random_state=42)
    # 提取训练集增强特征
    print('提取训练集增强特征...')
    X_train, y_train = extract_features_and_labels(train_imgs, train_masks)
    print('提取测试集增强特征...')
    X_test, y_test = extract_features_and_labels(test_imgs, test_masks)
    # 统计类别比例
    pos = np.sum(y_train == 1)
    neg = np.sum(y_train == 0)
    scale_pos_weight = neg / pos if pos > 0 else 1.0
    print(f"正样本: {pos}, 负样本: {neg}, scale_pos_weight: {scale_pos_weight:.2f}")
    # 训练
    bst, train_time = train_xgb(X_train, y_train, scale_pos_weight)
    # 保存模型
    os.makedirs('base_model_xgb', exist_ok=True)
    joblib.dump(bst, 'base_model_xgb/xgb_model.joblib')
    print('模型已保存到 base_model_xgb/xgb_model.joblib')
    # 预测
    y_pred, test_time = predict_xgb(bst, X_test)
    np.savez('base_model_xgb/xgb_pred.npz', y_pred=y_pred, y_test=y_test)
    print('预测结果已保存到 base_model_xgb/xgb_pred.npz')
    print(f'训练时间: {train_time:.2f}秒, 测试时间: {test_time:.2f}秒')

if __name__ == '__main__':
    main()