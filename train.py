import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from dataset import FaceDataset
from model.srcnn import SRCNN
from torchvision.models import vgg16

#Device (GPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

vgg = vgg16(pretrained=True).features[:16].eval().to(device)

for param in vgg.parameters():
    param.requires_grad = False

#dataset
dataset = FaceDataset("data/high_res")
dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

#model
model = SRCNN().to(device)

#loss function
def perceptual_loss(output, target):
    return nn.functional.mse_loss(vgg(output), vgg(target))
#criterion = nn.MSELoss()
#criterion = nn.L1Loss()

#optimizer
optimizer = optim.Adam(model.parameters(), lr=1e-4)

#training loop
epochs = 15

for epoch in range(epochs):
    total_loss = 0

    for lr_img, hr_img in dataloader:
        lr_img = lr_img.to(device)
        hr_img = hr_img.to(device)

        #forward pass
        output = model(lr_img)

        #loss
        #loss = criterion(output, hr_img)
        loss = 0.6 * nn.L1Loss()(output, hr_img) + 0.4 * perceptual_loss(output, hr_img)

        #backprop
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch [{epoch+1}/{epochs}], Loss: {total_loss:.4f}")

#save model
torch.save(model.state_dict(), "srcnn.pth")