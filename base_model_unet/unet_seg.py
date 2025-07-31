import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# ----------- U-Net网络结构 -----------
class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1):
        super().__init__()
        def CBR(in_c, out_c):
            return nn.Sequential(
                nn.Conv2d(in_c, out_c, 3, padding=1),
                nn.BatchNorm2d(out_c),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_c, out_c, 3, padding=1),
                nn.BatchNorm2d(out_c),
                nn.ReLU(inplace=True),
            )
        self.enc1 = CBR(in_channels, 64)
        self.enc2 = CBR(64, 128)
        self.enc3 = CBR(128, 256)
        self.enc4 = CBR(256, 512)
        self.pool = nn.MaxPool2d(2)
        self.center = CBR(512, 1024)
        self.up4 = nn.ConvTranspose2d(1024, 512, 2, stride=2)
        self.dec4 = CBR(1024, 512)
        self.up3 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.dec3 = CBR(512, 256)
        self.up2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec2 = CBR(256, 128)
        self.up1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec1 = CBR(128, 64)
        self.final = nn.Conv2d(64, out_channels, 1)
    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        c = self.center(self.pool(e4))
        d4 = self.dec4(torch.cat([self.up4(c), e4], 1))
        d3 = self.dec3(torch.cat([self.up3(d4), e3], 1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], 1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], 1))
        out = self.final(d1)
        return out

# ----------- 数据集 -----------
class SegDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        img = self.X[idx].transpose(2,0,1)  # HWC->CHW
        mask = self.y[idx][None, ...]       # H,W->1,H,W
        return torch.tensor(img, dtype=torch.float32), torch.tensor(mask, dtype=torch.float32)

# ----------- 训练与预测 -----------
def train_unet():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    X_train = np.load('base_model_unet/X_train.npy')
    y_train = np.load('base_model_unet/y_train.npy')
    X_test = np.load('base_model_unet/X_test.npy')
    y_test = np.load('base_model_unet/y_test.npy')
    train_ds = SegDataset(X_train, y_train)
    test_ds = SegDataset(X_test, y_test)
    train_loader = DataLoader(train_ds, batch_size=8, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=8, shuffle=False)
    model = UNet().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.BCEWithLogitsLoss()
    # 训练
    for epoch in range(1, 16):
        model.train()
        epoch_loss = 0
        for imgs, masks in tqdm(train_loader, desc=f'Epoch {epoch}', leave=False):
            imgs, masks = imgs.to(device), masks.to(device)
            optimizer.zero_grad()
            logits = model(imgs)
            loss = criterion(logits, masks)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * imgs.size(0)
        print(f'Epoch {epoch} loss: {epoch_loss/len(train_ds):.4f}')
    torch.save(model.state_dict(), 'base_model_unet/unet_model.pth')
    print('模型已保存到 base_model_unet/unet_model.pth')
    # 预测
    model.eval()
    preds = []
    with torch.no_grad():
        for imgs, _ in tqdm(test_loader, desc='Predict'):
            imgs = imgs.to(device)
            logits = model(imgs)
            probs = torch.sigmoid(logits)
            preds.append(probs.cpu().numpy())
    y_pred = np.concatenate(preds, axis=0)
    y_pred = y_pred[:,0]  # (N,1,H,W)->(N,H,W)
    np.save('base_model_unet/y_pred.npy', y_pred)
    print('预测结果已保存到 base_model_unet/y_pred.npy')

if __name__ == '__main__':
    train_unet() 