import torch
import torch.nn as nn
import cv2
import numpy as np
import os

MODEL_PATH = "models/unet_breast_model.pth"

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)

class UNet(nn.Module):

    def __init__(self):

        super().__init__()

        self.inc = DoubleConv(3,64)

        self.down1 = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(64,128)
        )

        self.down2 = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(128,256)
        )

        self.down3 = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(256,512)
        )

        self.bottle = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(512,1024)
        )

        self.up1 = nn.ConvTranspose2d(1024,512,2,2)
        self.conv_up1 = DoubleConv(1024,512)

        self.up2 = nn.ConvTranspose2d(512,256,2,2)
        self.conv_up2 = DoubleConv(512,256)

        self.up3 = nn.ConvTranspose2d(256,128,2,2)
        self.conv_up3 = DoubleConv(256,128)

        self.up4 = nn.ConvTranspose2d(128,64,2,2)
        self.conv_up4 = DoubleConv(128,64)

        self.outc = nn.Conv2d(64,1,1)

        self.sigmoid = nn.Sigmoid()

    def forward(self,x):

        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)

        xb = self.bottle(x4)

        x = self.up1(xb)
        x = torch.cat([x,x4],dim=1)
        x = self.conv_up1(x)

        x = self.up2(x)
        x = torch.cat([x,x3],dim=1)
        x = self.conv_up2(x)

        x = self.up3(x)
        x = torch.cat([x,x2],dim=1)
        x = self.conv_up3(x)

        x = self.up4(x)
        x = torch.cat([x,x1],dim=1)
        x = self.conv_up4(x)

        return self.sigmoid(self.outc(x))

model = UNet().to(device)

model.load_state_dict(
    torch.load(MODEL_PATH, map_location=device)
)

model.eval()

def create_mask(img_path):

    image = cv2.imread(img_path)

    image = cv2.resize(image,(256,256))

    x = image.astype(np.float32)/255.0

    x = np.transpose(x,(2,0,1))

    x = torch.tensor(x).unsqueeze(0).to(device)

    with torch.no_grad():

        pred = model(x)

        pred = (
            pred > 0.5
        ).float().cpu().numpy()[0][0]

    mask = (pred * 255).astype(np.uint8)

    filename = os.path.basename(img_path)

    save_path = f"outputs/masks/{filename}"

    cv2.imwrite(save_path, mask)

    return save_path