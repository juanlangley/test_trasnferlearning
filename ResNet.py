
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
NUM_TRAIN = 45000
MINIBATCH_SIZE = 64

transform_cifar = T.Compose([
                T.ToTensor(),
                T.Normalize([0.491, 0.482, 0.447], [0.247, 0.243, 0.261])
            ])

# Training set loader
cifar10_train = datasets.CIFAR10(DATA_PATH, train=True, download=True,
                             transform=transform_cifar)
train_loader = DataLoader(cifar10_train, batch_size=MINIBATCH_SIZE, 
                          sampler=sampler.SubsetRandomSampler(range(NUM_TRAIN)))

# Validation set loader
cifar10_val = datasets.CIFAR10(DATA_PATH, train=True, download=True,
                           transform=transform_cifar)
val_loader = DataLoader(cifar10_val, batch_size=MINIBATCH_SIZE, 
                        sampler=sampler.SubsetRandomSampler(range(NUM_TRAIN, len(cifar10_val))))

# Testing set loader
cifar10_test = datasets.CIFAR10(DATA_PATH, train=False, download=True, 
                            transform=transform_cifar)
test_loader = DataLoader(cifar10_test, batch_size=MINIBATCH_SIZE)


if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

#ejemplo de iteración
#for i, (x,y) in enumerate(val_loader):
#    print(i,x.shape,y.shape)


classes = ["Avión", "Carro", "Ave", "Gato", "Venado", "Perro", "Rana", "Caballos", "Barco", "Camión"]

#Muestreo de imagenes EJEMPLO
def plot_figure(image):
    plt.imshow(image.permute(1,2,0))
    plt.axis("off")
    plt.show()
    
rnd_sample_idx = np.random.randint(len(test_loader))
print("La imagen muestreada representa un: {}".format(classes[test_loader.dataset[rnd_sample_idx][1]]))

image = test_loader.dataset[rnd_sample_idx][0]
image = (image - image.min()) / (image.max() - image.min())
plot_figure(image)


# CALCULAR ACCURACY

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
    
# RESNET CARGAR MODELO PRE-CARGADO - Modelo preentrenado
model_resnet18 = models.resnet18(pretrained=True)

#explorar modelo resnet

for i,w in enumerate(model_resnet18.parameters()):
    print(i,w.shape, w.requires_grad) #requires_grad es que va a ser reentrenado

# AJUSTAR MODELO - no se entrena ya que lleva mucho tiempo

#creamos modelo auxiliar
#model_aux = nn.Sequential(*list(model_resnet18.children())) 

#cargamos sin la ultima capa (es la que tiene la salida)
model_aux = nn.Sequential(*list(model_resnet18.children())[:-1])

#ponemos para q no se entrene
for i, parameter in enumerate(model_aux.parameters()):
    parameter.requires_grad = False

#for i, parameter in enumerate(model_aux.parameters()):
#    print(i, parameter.requires_grad)



# LOOP DE ENTRENAMIENTO

def train(model, optimiser, epochs=100):
#     def train(model, optimiser, scheduler = None, epochs=100):
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


hidden1 = 256 
hidden = 256
lr = 5e-4
epochs = 20

#Sin RESNET

model1 = nn.Sequential(nn.Flatten(),
                        nn.Linear(in_features=32*32*3, out_features=hidden1), 
                        nn.ReLU(),
                        nn.Linear(in_features=hidden1, out_features=hidden), 
                        nn.ReLU(),
                        nn.Linear(in_features=hidden, out_features=10))

#CON RESNET

model1 = nn.Sequential(model_aux,
                       nn.Flatten(), 
                       nn.Linear(in_features=512, out_features= 10, bias= True))

#model2
optimiser = torch.optim.Adam(model1.parameters(), lr=lr, betas=(0.9, 0.999))

train(model1, optimiser, epochs)


accuracy(model1, test_loader)





