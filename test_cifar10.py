import torch
import torchvision
import torchvision.transforms as transforms
from models.ResNet import resnet26, resnet8

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("device:", device)

model = resnet26('max_accuracy.pth').to(device)

#model = resnet8()
#model.load_state_dict(torch.load('resnet8_cifar10.pth'))
#model.to(device)

batch_size = 128

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=0)

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