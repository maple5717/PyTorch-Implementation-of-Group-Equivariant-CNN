import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch
import torch.nn.init as init
import math
from resnet import ResNet
from tqdm import tqdm
import numpy as np

transform = transforms.Compose(
    [
     transforms.ToTensor(),
    ])

trainset = torchvision.datasets.CIFAR10(root='./cifar10', train=True,
                                        download=True, transform=transform)
trainloader = DataLoader(trainset, batch_size=128, shuffle=True)

testset = torchvision.datasets.CIFAR10(root='./cifar10', train=False,
                                       download=True, transform=transform)
testloader = DataLoader(testset, batch_size=128, shuffle=False)


from tqdm.notebook import tqdm
import numpy as np

# Define the device to use for training (GPU if available, else CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define the transformation to be applied to the input images



# Define the ResNet model
structure = [(64, 1), (64, 1), (128, 2), (128, 2), (256, 2), (256, 2)]
# structure = [(16, 1)]

use_gcnn = True
model = ResNet(structure, use_gcnn)

text = "gcnn" if use_gcnn else ""

# Define the loss function (cross-entropy)
criterion = torch.nn.CrossEntropyLoss()

# Define the optimizer (e.g. stochastic gradient descent)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)

# Move the model to the device
model = model.to(device)

# Train the model
print("start training")
train_accuracy = []
val_accuracy = []
for epoch in range(75):
    running_loss = 0.0
    correct = 0
    total = 0
    model.train()
    for i, data in (enumerate(tqdm(trainloader), 0)):
        inputs, labels = data
        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()


        running_loss += loss.item()
        
        _, predicted = torch.max(outputs.data, 1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)
       
    train_acc = correct / total
    print('Epoch %d Accuracy of the network on the training images: %d %%' % (
    epoch+1, 100 * train_acc))
    train_accuracy.append(train_acc)
    np.save(f'acc_train_{text}.npy', np.array(train_accuracy))

    # Evaluate the model on the test set
    correct = 0
    total = 0
    model.eval()
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    val_acc = correct / total
    print('Accuracy of the network on the test images: %d %%' % (
        100 * val_acc))
    val_accuracy.append(val_acc)
    torch.save(model.state_dict(), f"mymodel_{text}.pth")
    np.save(f'acc_val_{text}.npy', np.array(val_accuracy))