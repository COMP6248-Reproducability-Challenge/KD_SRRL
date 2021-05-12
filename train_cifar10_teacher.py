import torch
import torchvision
import torchvision.transforms as transforms
from models.ResNet import ResNet
import torch.optim as optim
import torch.nn as nn
import time
from utils.utils import lr_step_policy

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("device:", device)

transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

# Normalize the test set same as training set without augmentation
transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

batch_size = 128

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=0)

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=0)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

model = ResNet(26, 10, bottleneck=False).to(device)  # ResNet-26
#model = ResNet(14, 10, bottleneck=False).to(device)  # ResNet-14
#model = nn.DataParallel(ResNet(26, 10, bottleneck=False)).to(device)  # ResNet-26 for multiple GPU, but there will be some small problems when loading the model

criterion = nn.CrossEntropyLoss()

#optimizer = optim.Adam(model.parameters())
optimizer = optim.SGD(model.parameters(), lr=0.1)

lr_scheduler = lr_step_policy(0.1, [150, 250, 320], 0.1, 0)

# start training
print("training start!")
time_start = time.time()
max_accuracy = 0
for epoch in range(350):
    model.train()
    lr_scheduler(optimizer, epoch)
    running_loss = 0.0
    for i, data in enumerate(trainloader):
        inputs, labels = data[0].to(device), data[1].to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 100 == 99:    # print every 100 mini-batches
            print('[%d, %5d] loss: %.5f' % (epoch + 1, i + 1, running_loss / 100))
            running_loss = 0.0

    # after every epoch, test the model
    correct = 0
    total = 0
    model.eval()
    with torch.no_grad():
        for data in testloader:
            images, labels = data[0].to(device), data[1].to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        accuracy = correct / total
        print("epoch:", epoch, "test accuracy:", accuracy, "time:", (time.time() - time_start) / 60, 'mins')
        if accuracy > max_accuracy:
            max_accuracy = accuracy
            torch.save(model.state_dict(), 'max_accuracy.pth')

print("training is over, total training time:", (time.time() - time_start) / 60, 'mins')

# save the final model
PATH = 'resnet26_cifar10.pth'
torch.save(model.state_dict(), PATH)

# test the accuracy
correct = 0
total = 0
model.eval()
with torch.no_grad():
    for data in testloader:
        images, labels = data[0].to(device), data[1].to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print('Accuracy of the network on the 10000 test images: %5f %%' % (100 * correct / total))