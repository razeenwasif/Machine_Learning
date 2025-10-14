import torch.nn as nn
import torch.nn.functional as F 

class CNN(nn.Module):
    def __init__(self, dropout_rate, fc1_neurons):
        super(CNN, self).__init__()
        
        # Convolutional layer
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.bn1 = nn.BatchNorm2d(10)
        self.bn2 = nn.BatchNorm2d(20)
        self.conv2_drop = nn.Dropout2d()
        # Fully connected layers
        self.fc1 = nn.Linear(320, fc1_neurons)
        self.fc2 = nn.Linear(fc1_neurons, 60)
        self.dropout_rate = dropout_rate

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.bn1(self.conv1(x)), kernel_size=2))
        x = F.relu(F.max_pool2d(self.bn2(self.conv2_drop(self.conv2(x))), kernel_size=2))
        # Flatten data
        x = x.view(-1, 320) #x.view(x.size(0), -1) # x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=self.dropout_rate, training=self.training)
        x = self.fc2(x)

        return F.log_softmax(x, dim=1)# x
 
