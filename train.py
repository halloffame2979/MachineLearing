import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from mnist import MNIST
import numpy as np

# Change info
batch_size = 100
learning_rate = 0.1
epochs = 10
path = "./Clothes dataset"
# Load train data
data_load = MNIST(path)
data_load.load_training()

X = np.asarray(data_load.train_images)/255 
y = np.asarray(data_load.train_labels)

X = torch.from_numpy(X).type(torch.FloatTensor)
y = torch.from_numpy(y).type(torch.LongTensor)

input_size = X.size()[1]
output_size = torch.unique(y).size()[0]

train = torch.utils.data.TensorDataset(X,y)

train_loader = torch.utils.data.DataLoader(train, batch_size=batch_size)

# Module neural network

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

model = ClassificationModule()


#loss function
criterion = nn.CrossEntropyLoss()

#optimizer
opt = optim.SGD(model.parameters(), lr=learning_rate)

for i in range(epochs):
    for data in train_loader:
        img, label = data
        img = img.view(batch_size, 1, 28, 28)
        opt.zero_grad()
        output = model(img)

        loss = criterion(output, label)
        
        
        loss.backward()
        opt.step()

torch.save(model.state_dict(), "state_dict.pth")

            
