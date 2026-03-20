from dataset import FaceDataset

dataset = FaceDataset("data/high_res")

lr, hr = dataset[0]

print(lr.shape, hr.shape)