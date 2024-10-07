
import numpy as np
from matplotlib import pyplot as plt

import torch
torch.manual_seed(0)

from torch import optim, nn
import torch.nn.functional as F
import torchvision.transforms as T

from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import models, datasets


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



DATA_PATH = "DataBases/cifar-10-batches-py"
TRAIN_SIZE = 50000
VAL_SIZE = 5000
TEST_SIZE = 5000
MINIBATCH_SIZE = 512

transform_cifar10_train = T.Compose([
    #Darle variabilidad a las imagenes para que no sean siempre iguales
    T.RandomHorizontalFlip(p=0.3),
    T.ColorJitter(brightness=0.1, contrast=0.1, hue=0.05),
    T.RandomApply([T.RandomRotation(10), T.Resize(40), T.CenterCrop(32)]),
    
    T.ToTensor(),
    T.Normalize([0.491, 0.482, 0.447], [0.247, 0.243, 0.261]) #Valores estandar de cifar
    ])

transform_cifar10_test= T.Compose([
    T.ToTensor(),
    T.Normalize([0.491, 0.482, 0.447], [0.247, 0.243, 0.261]) #Valores estandar de cifar
    ])



# Training set loader
cifar10_train = datasets.CIFAR10(DATA_PATH, train=True, download=True,
                             transform=transform_cifar10_train)
train_loader = DataLoader(cifar10_train, batch_size=MINIBATCH_SIZE, shuffle=True)
# shuffle => los genera de forma aleatoria

# Validation and test set loader
test_dataset = datasets.CIFAR10(DATA_PATH, train=False, download=True, 
                            transform = transform_cifar10_test)
val_dataset, test_dataset = random_split(test_dataset, [VAL_SIZE, TEST_SIZE])

val_loader = DataLoader(val_dataset, batch_size= MINIBATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size = MINIBATCH_SIZE, shuffle=True)


#len(test_loader.dataset)
#for i,(x,y) in enumerate(test_loader):
#    print(i, x.shape, y.shape)

classes = train_loader.dataset.classes

# ------------------------------------
#Muestreo de imagenes EJEMPLO
def plot_figure(image):
    plt.figure(figsize=(3,3))
    plt.imshow(image.permute(1,2,0))
    plt.axis("off")
    plt.show()
    
rnd_sample_idx = np.random.randint(len(test_loader))
print("La imagen muestreada representa un: {}".format(classes[test_loader.dataset[rnd_sample_idx][1]]))

image = test_loader.dataset[rnd_sample_idx][0]
image = (image - image.min()) / (image.max() - image.min())
plot_figure(image)


# ------------------------------------
#PONER IMAGENES EN COLUMNAS
def plot_cifar10_grid():
    classes = train_loader.dataset.classes
    total_samples = 8
    plt.figure(figsize=(15,15))
    for label, sample in enumerate(classes):
        class_idxs = np.flatnonzero(label == np.array(train_loader.dataset.targets))
        sample_idxs = np.random.choice(class_idxs, total_samples, replace = False)
        for i, idx in enumerate(sample_idxs):
            plt_idx = i*len(classes) + label + 1
            plt.subplot(total_samples, len(classes), plt_idx)
            plt.imshow(train_loader.dataset.data[idx])
            plt.axis('off')
            
            if i == 0: plt.title(sample)
    plt.show()

plot_cifar10_grid() 

# ------------------------------------


def accuracy(model, loader):
    correct = 0
    total = 0
    cost = 0
    model.eval()
    model = model.to(device=device)
    
    with torch.no_grad():
        for (x, y) in loader:
            x = x.to(device=device, dtype = torch.float32)
            y = y.to(device=device, dtype = torch.long)
            scores = model(x)
            cost += (F.cross_entropy(scores,y)).item()
            
            _, pred = scores.max(dim=1) #pred shape (mb_size )
            correct += (pred == y).sum() # pred shape (mb_size), yi shape (mb_size, 1)
            total += pred.size(0)
        return cost/len(loader),float(correct)/total 
    


# ------------------------------------
#BUSCAR LEARNING RATE


def find_lr(model, optimiser, start_val = 1e-6, end_val = 1, beta = 0.99, loader = train_loader):
    n = len(loader) - 1
    factor = (end_val / start_val)**(1/n)
    lr = start_val
    
    optimiser.param_groups[0]['lr'] = lr #this allows you to update the learning rate
    avg_loss, loss, acc = 0., 0., 0.
    lowest_loss = 0.
    batch_num = 0
    losses = []
    log_lrs = []
    accuracies = []
    model = model.to(device=device)
    for i, (x, y) in enumerate(loader, start=1):
        x = x.to(device = device, dtype = torch.float32)
        y = y.to(device = device, dtype = torch.long)
        optimiser.zero_grad()
        scores = model(x)
        cost = F.cross_entropy(input=scores, target=y)
        
        # momentum o promedio exponencial ponderado para nuestra perdida 
        #(momentum de beta) (para grafica sin tanto ruido)
        loss = beta*loss + (1-beta)*cost.item()
        #bias correction
        avg_loss = loss/(1 - beta**i)
        
        acc_ = ((torch.argmax(scores, dim=1) == y).sum()/scores.size(0)) 
        #promedio ponderado de acc
#         acc = beta*acc + (1-beta)*acc_.item()
#         avg_acc = acc/(1 - beta**i)
        #if loss is massive stop
        if i > 1 and avg_loss > 4 * lowest_loss:
            print(f'from here{i, cost.item()}')
            return log_lrs, losses, accuracies
        if avg_loss < lowest_loss or i == 1:
            lowest_loss = avg_loss
        
#         accuracies.append(acc.item())
        accuracies.append(acc_.item())
#         accuracies.append(avg_acc)
        losses.append(avg_loss)
        log_lrs.append(lr)
        #step
        cost.backward()
        optimiser.step()
        #update lr
        print(f'cost:{cost.item():.4f}, lr: {lr:.4f}, acc: {acc_.item():.4f}')
        lr *= factor
        optimiser.param_groups[0]['lr'] = lr
        
    return log_lrs, losses, accuracies     

# ------------------------------------
#scheduler - objeto de pytorch que permite actualizar el lr a traves del proceso de entrenamiento
def train(model, optimiser, scheduler = None, epochs = 100):
    model = model.to(device = device)
    #guardamos los valores para poder graficarlos
    val_loss_history = []
    train_loss_history = []
    val_acc_history = []
    train_acc_history = []
    lrs = []
    
    train_cost = 0.
    val_cost = 0.
    train_cost_acum = 0
    for epoch in range(epochs):
        train_correct_num  = 0
        train_total = 0
        train_cost_acum = 0
        for mb, (x, y) in enumerate(train_loader, start=1):
            model.train()
            x = x.to(device=device, dtype=torch.float32)
            y = y.to(device=device, dtype=torch.long)
            scores = model(x)
            cost = F.cross_entropy(input=scores, target=y)
            optimiser.zero_grad()
            cost.backward()
            optimiser.step()
            #if using scheduler
            #nos permite hacer un paso del objeto sheduler para
            #actualizar el lr de acuerdo de la politica, del one cycle policy
            if scheduler: scheduler.step()
            
            #acumular los valores
            train_correct_num += (torch.argmax(scores, dim=1) == y).sum()
            train_total += scores.size(0)        
            train_cost_acum += cost.item()
            train_acc = float(train_correct_num)/train_total  
#             train_cost = train_cost_acum/mb
            val_cost, val_acc = accuracy(model, val_loader)

            val_loss_history.append(val_cost)
            train_loss_history.append(cost.item())
            val_acc_history.append(val_acc)
            train_acc_history.append(train_acc)
#             lrs.append(scheduler.get_last_lr()[0])
            lrs.append(optimiser.param_groups[0]["lr"])
        
        #f'last lr: {scheduler.get_last_lr()[0]:6f},
        
        train_acc = float(train_correct_num)/train_total
        train_cost = train_cost_acum/len(train_loader)
        print(f'Epoch:{epoch}, train cost: {train_cost:.6f}, val cost: {val_cost:.6f},'
                      f' train acc: {train_acc:.4f}, val acc: {val_acc:4f}, total: {train_total},'
                      f' lr: {optimiser.param_groups[0]["lr"]:.6f}')
        
        #guardar modelo si se quiere
        """
        if epoch%10 ==0:
            torch.save({
                "epoch" : epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimiser.state_dict(),
                "loss":csot,
                }, f"/models/check{epoch}.pt")
        """
        
    return train_loss_history, val_loss_history, train_acc_history, val_acc_history, lrs


# ------------------------------------

# ResNet56
conv_k_3 = lambda channel1, channel2, stride: nn.Conv2d(channel1, channel2, stride = stride, kernel_size=3, padding=1)

class residual_block(nn.Module):
    def __init__(self, in_channel, out_channel, stride=1, change_size=True):
        super(residual_block, self).__init__()
        self.conv1 = conv_k_3(in_channel, out_channel, stride)
        self.bn1 = nn.BatchNorm2d(out_channel)
        self.conv2 = conv_k_3(out_channel, out_channel, 1)
        self.bn2 = nn.BatchNorm2d(out_channel)
        self.change_size = change_size
        
        if change_size:
            self.residual= nn.Sequential(nn.Conv2d(in_channel, out_channel, kernel_size=1, stride=stride),
                                         nn.BatchNorm2d(out_channel)                                         
                                         )
        
    def forward(self,x):
        identity = x if not self.change_size else self.residual(x)
        y = F.relu(self.bn1(self.conv1(x)))
        y = self.bn2(self.conv2(y))
        y += identity
        return F.relu(y)


class ResNet56(nn.Module):
    def __init__(self, n=9, num_classes=10):
        super(ResNet56, self).__init__()
        #valores por defecto
        self.conv1 = conv_k_3(3, 16, stride=1)
        self.bn1 = nn.BatchNorm2d(16)
        
        self.block1 = self.create_block(n=9, in_channel=16, 
                                        out_channel=16, stride=1, 
                                        change_size=False)
        self.block2 = self.create_block(n=9, in_channel=16, 
                                        out_channel=32, stride=2, 
                                        change_size=True)
        self.block3 = self.create_block(n=9, in_channel=32,
                                        out_channel=64, stride=2,
                                        change_size=True)
        
        self.fc = nn.Linear(64, num_classes)
        
    def create_block(self, n, in_channel, out_channel, stride, change_size=True):
        block = [residual_block(in_channel, out_channel, stride, change_size)]
        for i in range(n-1):
            block.append(residual_block(out_channel, out_channel,stride=1, change_size=False))
        return nn.Sequential(*block)
        
    def forward(self, x):
        y = F.relu(self.bn1(self.conv1(x)))
        y = self.block3(self.block2(self.block1(y)))
        y = F.adaptive_avg_pool2d(y,1)
        return self.fc(y.view(y.size(0), -1))



model_resnet56 = ResNet56()
optimiser_resnet56 = torch.optim.SGD(model_resnet56.parameters(), 
                                     lr=0.1, momentum=0.95,
                                     weight_decay=1e-4)
#weight_decay => regularización conocida como l2 -


lg_lr, losses, accuracies = find_lr(model_resnet56, optimiser_resnet56,
                                    start_val=1e-6, end_val=10)

#grafico de lr
f1, ax1 = plt.subplots(figsize=(20,10))
# ax1.plot(lr[60:-2], losses[60:-2])
ax1.plot(lg_lr, losses)
ax1.set_xscale('log')
ax1.set_xticks([1e-1,2e-1, 1, 10])
ax1.get_xaxis().get_major_formatter().labelOnlyBase = False
plt.show()


#Accuracy
f1, ax1 = plt.subplots(figsize=(20,10))
# ax1.plot(lr[60:-2], losses[60:-2])
ax1.plot(lg_lr, accuracies)
ax1.set_xscale('log')
# ax1.set_xticks([1e-1, 2e-1,5e-1, 7e-1, 1, 10])
ax1.get_xaxis().get_major_formatter().labelOnlyBase = False
plt.show()


#Modelo con el lr
model_resnet56 = ResNet56()
optimiser_resnet56 = torch.optim.SGD(model_resnet56.parameters(),
                                     lr=0.1, momentum=0.95,
                                     weight_decay=1e-4)
epochs = 20
scheduler = torch.optim.lr_scheduler.OneCycleLR(optimiser_resnet56, 
                                                max_lr=2e-1, 
                                                steps_per_epoch=len(train_loader), 
                                                epochs = epochs, pct_start=0.43, 
                                                div_factor=10, 
                                                final_div_factor=1000, 
                                                three_phase=True, verbose=False
                                            )
train_loss_history, val_loss_history, train_acc_history, val_acc_history, lrs = train(
                                model_resnet56, 
                                optimiser=optimiser_resnet56,
                                scheduler=scheduler,
                                epochs = epochs
                                )



accuracy(model_resnet56, test_loader)













