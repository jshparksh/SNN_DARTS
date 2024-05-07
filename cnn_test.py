import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from quantization import *
import utils
# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define transforms for the data
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

# Load CIFAR-10 dataset
trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=32,
                                          shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=32,
                                         shuffle=False, num_workers=2)

# Define the CNN architecture
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv11 = nn.Conv2d(32, 32, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pact1 = PACT_with_log_quantize()
        self.pact11 = PACT_with_log_quantize()
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv22 = nn.Conv2d(64, 64, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pact2 = PACT_with_log_quantize()
        self.pact22 = PACT_with_log_quantize()
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv33 = nn.Conv2d(128, 128, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pact3 = PACT_with_log_quantize()
        self.pact33 = PACT_with_log_quantize()
        self.conv4 = nn.Conv2d(128, 256, 3, padding=1)
        self.conv44 = nn.Conv2d(256, 256, 3, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        self.pact4 = PACT_with_log_quantize()
        self.pact44 = PACT_with_log_quantize()
        self.conv5 = nn.Conv2d(256, 512, 3, padding=1)
        self.conv55 = nn.Conv2d(512, 512, 3, padding=1)
        self.bn5 = nn.BatchNorm2d(512)
        self.pact5 = PACT_with_log_quantize()
        self.pact55 = PACT_with_log_quantize()
        self.conv6 = nn.Conv2d(512, 512, 3, padding=1)
        self.bn6 = nn.BatchNorm2d(512)
        self.pact6 = PACT_with_log_quantize()
        self.pool1 = nn.MaxPool2d(2, 2)
        self.pool2 = nn.MaxPool2d(2, 1)
        self.fc1 = nn.Linear(512, 1024)
        self.fc2 = nn.Linear(1024, 10)
    def forward(self, x):
        x = self.pool1(self.pact1(self.bn1(self.conv1(x))))
        x = self.pact11(self.bn1(self.conv11(x)))
        x = self.pool2(self.pact2(self.bn2(self.conv2(x))))
        x = self.pact22(self.bn2(self.conv22(x)))
        x = self.pool1(self.pact3(self.bn3(self.conv3(x))))
        x = self.pact33(self.bn3(self.conv33(x)))
        x = self.pool2(self.pact4(self.bn4(self.conv4(x))))
        x = self.pact44(self.bn4(self.conv44(x)))
        x = self.pool1(self.pact5(self.bn5(self.conv5(x))))
        x = self.pact55(self.bn5(self.conv55(x)))
        x = self.pool1(self.pact6(self.bn6(self.conv6(x))))
        #x = self.pool(torch.relu(self.bn1(self.conv1(x))))
        #x = self.pool(torch.relu(self.bn2(self.conv2(x))))
        #x = self.pool(torch.relu(self.bn3(self.conv3(x))))
        x = x.view(-1, 512)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    
# Initialize the network
net = SimpleCNN().to(device)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

# Training the network
utils.base_mode_switch(net)
for epoch in range(1000):  # loop over the dataset multiple times

    running_loss = 0.0
    correct = 0
    total = 0
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data[0].to(device), data[1].to(device)

        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        if i % 200 == 199:  # print every 200 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 200))
            running_loss = 0.0
            base, _ = utils.print_base(net, [])
            base_grad, _ = utils.print_base_grad(net, [])
            for i in range(len(base)):
                print(base[i][0], base[i][1], base_grad[i][1])
    epoch_accuracy = 100 * correct / total
    print('Epoch %d: Accuracy on the training images: %d %%' % (epoch + 1, epoch_accuracy))
    
    # Test the network on the test data
    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            inputs, labels = data[0].to(device), data[1].to(device)
            outputs = net(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print('Accuracy of the network on the 10000 test images: %d %%' % (100 * correct / total))
print('Finished Training')




