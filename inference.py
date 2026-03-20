import torch
import cv2
import matplotlib.pyplot as plt
from torchvision import transforms

from model.srcnn import SRCNN

#load model
model = SRCNN()
model.load_state_dict(torch.load("srcnn.pth", map_location="cpu"))
model.eval()

transform = transforms.ToTensor()

#load test image
img = cv2.imread("test.jpg")
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

#resize
img = cv2.resize(img, (128, 128))

#create low-res version
#lr = cv2.resize(img, (64, 64))
#lr = cv2.resize(lr, (128, 128))

#convert to tensor
lr_tensor = transform(img).unsqueeze(0)

#inference
with torch.no_grad():
    output = model(lr_tensor)

#convert output to image
output_img = output.squeeze(0).permute(1, 2, 0).numpy()
output_img = (output_img * 0.5) + 0.5
output_img = output_img.clip(0,1)

#plot
plt.figure(figsize = (10, 5))

plt.subplot(1, 2, 1)
plt.title("Low Resolution")
plt.imshow(img)

plt.subplot(1, 2, 2)
plt.title("Enhanced")
plt.imshow(output_img)

plt.show()