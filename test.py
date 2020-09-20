import numpy as np
import pandas as pd 
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from mnist import MNIST

# Change info
batch_size = 100
path = "./Dataset/Clothes dataset"

# Load test data
data_load = MNIST(path)
data_load.load_testing()

X = np.asarray(data_load.test_images)/255 
y = np.asarray(data_load.test_labels)

X = torch.from_numpy(X).type(torch.FloatTensor)
y = torch.from_numpy(y).type(torch.LongTensor)

test = torch.utils.data.TensorDataset(X,y)

test_loader = torch.utils.data.DataLoader(test, batch_size=batch_size)

class ClassificationModule(nn.Module):
    def __init__(self):
        super(ClassificationModule, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5, 1)
        self.conv2 = nn.Conv2d(6, 16, 5, 1)
        self.linear1 = nn.Linear(16*4*4,50)
        self.linear2 = nn.Linear(50,10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.max_pool2d(x, 2)
        x = self.conv2(x)
        x = F.max_pool2d(x, 2)
        x = x.view(-1,16*4*4)
        x = self.linear1(x)
        x = self.linear2(x)
        return x


PATH = "state_dict.pth"

model = ClassificationModule()
model.load_state_dict(torch.load(PATH))
model.eval()


correct = 0
total = 0
with torch.no_grad():
    for data in test_loader:
        images, labels = data
        images = images.view(batch_size, 1, 28, 28)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print('Accuracy of the network on the',total, 'test images: %d %%' % (
    100 * correct / total))

