
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import DataLoader
from torch.utils.data import sampler
import torchvision.datasets as datasets
import torchvision.transforms as T
from torchvision import models
from matplotlib import pyplot as plt



DATA_PATH = "DataBases/cifar-10-batches-py"
NUM_TRAIN = 50000
NUM_VAL = 5000
NUM_TEST = 5000
MINIBATCH_SIZE = 64

transform_cifar = T.Compose([
    T.ToTensor(),
    T.Normalize([0.491, 0.482, 0.447], [0.247, 0.243, 0.261]) #Valores estandar de cifar
    ])

# Train dataset
cifar10_train = datasets.CIFAR10(DATA_PATH, train=True, download=True,
                             transform=transform_cifar)

train_loader = DataLoader(cifar10_train, batch_size=MINIBATCH_SIZE, 
                          sampler=sampler.SubsetRandomSampler(range(NUM_TRAIN)))
#Validation set
cifar10_val = datasets.CIFAR10(DATA_PATH, train=False, download=True,
                           transform=transform_cifar)

val_loader = DataLoader(cifar10_val, batch_size=MINIBATCH_SIZE, 
                        sampler=sampler.SubsetRandomSampler(range(NUM_VAL)))
#Test set
cifar10_test = datasets.CIFAR10(DATA_PATH, train=False, download=True, 
                            transform=transform_cifar)

test_loader = DataLoader(cifar10_test, batch_size=MINIBATCH_SIZE,
                        sampler=sampler.SubsetRandomSampler(range(NUM_VAL, len(cifar10_test))))

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")


classes = test_loader.dataset.classes


#Mostar Imagenes
def plot_figure(image):
    plt.imshow(np.transpose(image,(1,2,0)))
    plt.axis('off')
    plt.show()

rnd_sample_idx = np.random.randint(len(test_loader))
print(f'La imagen muestreada representa un: {classes[test_loader.dataset[rnd_sample_idx][1]]}')
image = test_loader.dataset[rnd_sample_idx][0]
image = (image - image.min()) / (image.max() -image.min() )
plot_figure(image)


# ------------------------------------
#PONER IMAGENES EN COLUMNAS
def plot_cifar10_grid():
    classes = test_loader.dataset.classes
    total_samples = 8
    plt.figure(figsize=(15,15))
    for label, sample in enumerate(classes):
        class_idxs = np.flatnonzero(label == np.array(test_loader.dataset.targets))
        sample_idxs = np.random.choice(class_idxs, total_samples, replace = False)
        for i, idx in enumerate(sample_idxs):
            plt_idx = i*len(classes) + label + 1
            plt.subplot(total_samples, len(classes), plt_idx)
            plt.imshow(test_loader.dataset.data[idx])
            plt.axis('off')
            
            if i == 0: plt.title(sample)
    plt.show()

plot_cifar10_grid() 
# ------------------------------------


def accuracy(model, loader):
    num_correct = 0
    num_total = 0
    model.eval()
    model = model.to(device=device)
    with torch.no_grad():
        for (xi, yi) in loader:
            xi = xi.to(device=device, dtype = torch.float32)
            yi = yi.to(device=device, dtype = torch.long)
            scores = model(xi) # mb_size, 10
            _, pred = scores.max(dim=1) #pred shape (mb_size )
            num_correct += (pred == yi).sum() # pred shape (mb_size), yi shape (mb_size, 1)
            num_total += pred.size(0)
        return float(num_correct)/num_total 
    


def train(model, optimiser, epochs=100):
    model = model.to(device=device)
    for epoch in range(epochs):
        for i, (xi, yi) in enumerate(train_loader):
            
            model.train()
            xi = xi.to(device=device, dtype=torch.float32)
            yi = yi.to(device=device, dtype=torch.long)
            scores = model(xi)

            cost = F.cross_entropy(input= scores, target=yi.squeeze())
        
            optimiser.zero_grad()           
            cost.backward()
            optimiser.step()           
            
        acc = accuracy(model, val_loader)
        #if epoch%5 == 0:     
        print(f'Epoch: {epoch}, costo: {cost.item()}, accuracy: {acc},')
#         scheduler.step()


# Sequential linear
hidden1 = 256
hidden = 256
lr = 0.0001
epochs = 10

model1 = nn.Sequential(nn.Flatten(),
                       nn.Linear(in_features=(32*32*3), out_features=hidden1),
                       nn.ReLU(),
                       nn.Linear(in_features=hidden1, out_features=hidden),
                       nn.ReLU(),
                       nn.Linear(in_features=hidden, out_features=10))

optimiser = torch.optim.Adam(model1.parameters(), lr=lr)
train(model1, optimiser,epochs)
# --------------------------------------------

# Sequential CNN
channel1 = 16
channel2 = 32
epochs = 10
lr = 0.0001

# el max pool quedaría: 
# tamaño de entrada es 32x32x3
# salida primera conv queda 32x32x16
# salida de la segunda conv 32x32x32
#el max pool vamos a tener 16x16x32 ((2,2) parte a la mitad)
modelCNN1 = nn.Sequential(nn.Conv2d(in_channels=3, out_channels=channel1, kernel_size=3, padding=1, bias=True),
                          nn.ReLU(),
                          nn.Conv2d(in_channels=channel1, out_channels=channel2, kernel_size=3, padding=1, bias=True),
                          nn.ReLU(),
                          nn.MaxPool2d(2,2),
                          nn.Flatten(),
                          nn.Linear(in_features=(16*16*channel2), out_features=10)
                          )

optimiser = torch.optim.Adam(modelCNN1.parameters(), lr=lr)
train(modelCNN1, optimiser,epochs)

# --------------------------------------------

# Orientado a objetos

class CNN_class1(nn.Module):
    def __init__(self, in_channel, channel1, channel2):
        super(CNN_class1, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels= channel1, 
                               kernel_size=3, padding = 1)
        self.conv2 = nn.Conv2d(in_channels=channel1, out_channels=channel2, 
                               kernel_size=3, padding=1)
        self.max_pool = nn.MaxPool2d(2,2)
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(in_features=(16*16*channel2), out_features=10)
        
    
    def forward(self,x):
        x = F.relu(self.conv2(F.relu(self.conv1(x))))
        x = self.max_pool(x)
        x = self.flatten(x)
        return self.fc(x)


channel1 = 16
channel2 = 32
epochs = 10
lr = 0.0001
modelCNN2 = CNN_class1(3, channel1,channel2)

optimiser = torch.optim.Adam(modelCNN2.parameters(), lr=lr)
train(modelCNN2, optimiser,epochs)


# --------------------------------------------

#Otra Forma de lo anterior
conv_k_3 = lambda channel1, channel2: nn.Conv2d(channel1, channel2, kernel_size=3, padding=1)

class CNN_class2(nn.Module):
    def __init__(self, in_channel, channel1, channel2):
        super(CNN_class2, self).__init__()
        
        self.conv1 = conv_k_3(in_channel, channel1)
        nn.init.kaiming_uniform_(self.conv1.weight)
        
        self.conv2 = conv_k_3(channel1, channel2)
        self.max_pool = nn.MaxPool2d(2,2)
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(in_features=(16*16*channel2), out_features=10)
        
    
    def forward(self,x):
        x = F.relu(self.conv2(F.relu(self.conv1(x))))
        x = self.max_pool(x)
        x = self.flatten(x)
        return self.fc(x)
    
    

channel1 = 16
channel2 = 32
epochs = 10
lr = 0.0001
modelCNN3 = CNN_class2(3, channel1,channel2)

optimiser = torch.optim.Adam(modelCNN3.parameters(), lr=lr)
train(modelCNN3, optimiser,epochs)

# --------------------------------------------

#Otra Forma con BATCHNORMALIZATION
conv_k_3 = lambda channel1, channel2: nn.Conv2d(channel1, channel2, kernel_size=3, padding=1)

class CNN_class3(nn.Module):
    def __init__(self, in_channel, channel1, channel2):
        super(CNN_class3, self).__init__()
        
        self.conv1 = conv_k_3(in_channel, channel1)
        self.bn1 = nn.BatchNorm2d(channel1)

        self.conv2 = conv_k_3(channel1, channel2)
        self.bn2 = nn.BatchNorm2d(channel2)
        
        self.max_pool = nn.MaxPool2d(2,2)
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(in_features=(16*16*channel2), out_features=10)
        
    
    def forward(self,x):
        x = F.relu(self.bn2(self.conv2(F.relu(self.bn1(self.conv1(x))))))
        x = self.max_pool(x)
        x = self.flatten(x)
        return self.fc(x)
    

channel1 = 16
channel2 = 32
epochs = 10
lr = 0.0001
modelCNN4 = CNN_class3(3, channel1,channel2)

optimiser = torch.optim.Adam(modelCNN4.parameters(), lr=lr)
train(modelCNN4, optimiser,epochs)


# --------------------------------------------



class CNN_class4(nn.Module):
    def __init__(self, in_channel, channel1, channel2):
        super(CNN_class4, self).__init__()
        
        self.conv1 = conv_k_3(in_channel, channel1)
        self.bn1 = nn.BatchNorm2d(channel1)

        self.conv2 = conv_k_3(channel1, channel2)
        self.bn2 = nn.BatchNorm2d(channel2)
        
        self.max_pool = nn.MaxPool2d(2,2)


    def forward(self,x):
        x = F.relu(self.bn2(self.conv2(F.relu(self.bn1(self.conv1(x))))))
        return self.max_pool(x)




channel1 = 16
channel2 = 32
channel3 = 64
channel4 = 128
epochs = 10
lr = 0.001

modelCNN5 = nn.Sequential(CNN_class4(3, channel1,channel2),
                          CNN_class4(channel2, channel3, channel4),
                          nn.Flatten(),
                          nn.Linear(in_features=8*8*channel4, out_features=10))


optimiser = torch.optim.Adam(modelCNN5.parameters(), lr=lr)
train(modelCNN5, optimiser,epochs)


accuracy(modelCNN5, test_loader)



