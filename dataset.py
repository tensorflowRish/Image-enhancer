import os
import cv2
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import numpy as np

class FaceDataset(Dataset):
    def __init__(self, image_dir, img_size=128, scale_factor=2, limit=5000):
        self.image_dir = image_dir
        self.image_list = os.listdir(image_dir)[:limit] #limited data
        self.img_size = img_size
        self.scale_factor = scale_factor

        self.transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5])
        ])

    def __len__(self):
        return len(self.image_list)
    
    def degrade_image(self, img):
        # Blur
        if np.random.rand() < 0.5:
            k = np.random.choice([3, 5])
            img = cv2.GaussianBlur(img, (k, k), 0)

        # Add noise
        if np.random.rand() < 0.5:
            noise = np.random.normal(0, 10, img.shape)
            img = img + noise
            img = np.clip(img, 0, 255).astype(np.uint8)

        return img
    
    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.image_list[idx])

        #read image
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        #resize to fixed size (high-res)
        #hr_img = cv2.resize(img, (self.img_size, self.img_size))
        
        #create low-res image
        # lr_img = cv2.resize(
        #     hr_img,
        #     (self.img_size//self.scale_factor, self.img_size//self.scale_factor)
        # )
        #lr_img = self.degrade_image(hr_img)

        #upscale back to original size (so input/output size match)
        #lr_img = cv2.resize(lr_img, (self.img_size, self.img_size))

        #convert to tensor
        # High-resolution image (target)
        hr_img = cv2.resize(img, (self.img_size * 2, self.img_size * 2))  # 256×256

        # Create low-resolution input
        lr_img = cv2.resize(
            hr_img,
            (self.img_size, self.img_size)  # 128×128
        )

        # Optional: apply degradation on LR
        lr_img = self.degrade_image(lr_img)

        hr_img = self.transform(hr_img)
        lr_img = self.transform(lr_img)

        return lr_img, hr_img
    
    
