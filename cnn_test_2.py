import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import utils
import torchvision.transforms as transforms
from quantization import *
# Define the CNN model
class CIFAR10_CNN(nn.Module):
    def __init__(self):
        super(CIFAR10_CNN, self).__init__()
        # Define the convolutional block
        def conv_block(in_channels, out_channels):
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                PACT_with_log_quantize(),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                PACT_with_log_quantize(),
                nn.MaxPool2d(kernel_size=2, stride=2)
            )
        
        def conv_no_reduce_block(in_channels, out_channels):
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                PACT_with_log_quantize(),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                PACT_with_log_quantize(),
            )
        # Define the linear block
        def linear_block(in_features, out_features):
            return nn.Sequential(
                nn.Linear(in_features, out_features),
                nn.BatchNorm1d(out_features),
                PACT_with_log_quantize(),
            )
        
        # Define the layers using Sequential
        self.features = nn.Sequential(
            conv_block(3, 32),
            conv_no_reduce_block(32, 32),
            conv_block(32, 64),
            conv_no_reduce_block(64, 64),
            conv_block(64, 128),
            conv_no_reduce_block(128, 128),
            conv_block(128, 256),
            conv_no_reduce_block(256, 256),
            conv_block(256, 512)
        )
        
        # Define the classifier
        self.classifier = nn.Sequential(
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            #PACT_with_log_quantize(),
            nn.Linear(512, 10)
        )

    def forward(self, x):
        x = self.features(x)
        # Flatten the output
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x
    
# Load CIFAR-10 dataset
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=32,
                                          shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=32,
                                         shuffle=False, num_workers=2)

# Checking if GPU is available and setting device accordingly
device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
torch.cuda.set_device(2)
print("Device:", device)

# Instantiate the model and move it to the GPU
net = CIFAR10_CNN().to(device)
alpha_params, base_params, model_params = utils.split_params(net)
# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
#optimizer = optim.SGD(model.parameters(), lr=0.025, momentum=0.9)
optimizer = optim.SGD(model_params, lr=0.025, momentum=0.9)
optimizer_alpha = optim.SGD(alpha_params, lr=1)
optimizer_base = optim.SGD(base_params, lr=1)

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 1000, eta_min=0.001)
scheduler_alpha = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_alpha, 1000, eta_min=0.001)
scheduler_base = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_base, 1000, eta_min=0.001)
# Training loop
utils.param_mode_switch(net)
for epoch in range(300):  # loop over the dataset multiple times
    running_loss = 0.0
    correct = 0
    total = 0
    scheduler.step()
    scheduler_base.step()
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data[0].to(device), data[1].to(device)

        optimizer.zero_grad()
        optimizer_alpha.zero_grad()
        optimizer_base.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer_base.step()
        optimizer_alpha.step()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        if i % 1 == 0:  # print every 200 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 200))
            running_loss = 0.0
            min_alpha, _ = utils.print_minimum_alpha(net, 1e6)
            print('min_alpha', min_alpha)
            alpha, _ = utils.print_alpha(net, [])
            alpha_grad, _ = utils.print_alpha_grad(net, [])
            base, _ = utils.print_base(net, [])
            base_grad, _ = utils.print_base_grad(net, [])
            print('-----------------alpha-----------------')
            for i in range(len(base)):
                print(alpha[i][0], alpha[i][1], alpha_grad[i][1])
            print('-----------------base-----------------')
            for i in range(len(base)):
                print(base[i][0], base[i][1], base_grad[i][1])
        
        utils.update_base(net, len(trainloader)-1)
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