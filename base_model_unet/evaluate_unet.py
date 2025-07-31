import numpy as np

def compute_iou(y_true, y_pred):
    intersection = np.logical_and(y_true, y_pred).sum()
    union = np.logical_or(y_true, y_pred).sum()
    if union == 0:
        return 1.0 if intersection == 0 else 0.0
    return intersection / union

def main():
    y_true = np.load('base_model_unet/y_test.npy')
    y_pred = np.load('base_model_unet/y_pred.npy')
    # 二值化（如果未二值化）
    y_pred_bin = (y_pred > 0.5).astype(np.uint8)
    iou = compute_iou(y_true, y_pred_bin)
    print(f'IoU (Jaccard) score: {iou:.4f}')

if __name__ == '__main__':
    main() 