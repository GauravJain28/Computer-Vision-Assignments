import os
import sys
import time
import random
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.nn import Linear, CrossEntropyLoss, Sequential
from torch.autograd import Variable

from tqdm import tqdm

root_dir = "./Assignment_4_dataset/group_1/"
seed = 51
BATCH_SIZE = 4
NUM_WORKERS = 2
wd = float(sys.argv[1]) # weight decay? # regularization
print("Weight Decay: {}".format(wd))

random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

stats = ((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))

train_transforms = transforms.Compose(
[
    transforms.ToTensor(), 
    transforms.Normalize(*stats,inplace=True)
])

test_transforms = transforms.Compose(
[
    transforms.ToTensor(), 
    transforms.Normalize(*stats,inplace=True)
])

train_imgs = torchvision.datasets.ImageFolder(root=os.path.join(root_dir,"train"), transform=train_transforms)
train_loader = DataLoader(train_imgs, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)

val_imgs = torchvision.datasets.ImageFolder(root=os.path.join(root_dir,"valid"), transform=test_transforms)
val_loader = DataLoader(val_imgs, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)

test_imgs = torchvision.datasets.ImageFolder(root=os.path.join(root_dir,"test"), transform=test_transforms)
test_loader = DataLoader(test_imgs, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

dir = os.path.join(root_dir,"train")
classes = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
classes.sort()

os.makedirs("./models", exist_ok=True)

vgg = torch.hub.load('pytorch/vision:v0.10.0', 'vgg16', pretrained=True)

for param in vgg.features.parameters():
    param.require_grad = True #FALSE

num_classes = len(classes)
num_feat = vgg.classifier[6].in_features
feat = list(vgg.classifier.children())
feat = feat[:-1]
feat.extend([Linear(num_feat, num_classes)])
vgg.classifier = Sequential(*feat)

if torch.cuda.is_available():
    vgg = vgg.cuda()

def train_model(x_train,y_train,model,optimizer,loss):
    tr_loss = 0
    x_train, y_train = Variable(x_train), Variable(y_train)
  
    if torch.cuda.is_available():     # converting the data into GPU format
        x_train = x_train.cuda()
        y_train = y_train.cuda()
        
    optimizer.zero_grad()             # zero the Gradients of parameters
    output_train = model(x_train)              # prediction for training and validation set
    loss_train = loss(output_train, y_train)   # computing the training and validation loss

    # computing the updated weights of all the model parameters
    loss_train.backward()
    optimizer.step()
    tr_loss = loss_train.item()

    return tr_loss

def predict_model(x_test,y_test,model):
    if torch.cuda.is_available():     # converting the data into GPU format
        x_test = x_test.cuda()
        y_test = y_test.cuda()

    with torch.no_grad():
      output = model(x_test)
    
    softmax = torch.exp(output).cpu()
    prob = list(softmax.numpy())
    predictions = np.argmax(prob, axis=1)
    
    return (np.sum(predictions==y_test.cpu().detach().numpy()))

print("------------ Training Begins ------------")
num_epochs = 15
lr = 0.001
loss = CrossEntropyLoss()

def calc_num_params(model):
    return np.sum([p.nelement() for p in model.parameters()])

num_params = calc_num_params(vgg)
print("Total model params: {}".format(num_params))


optimizer = torch.optim.SGD(vgg.parameters(), lr=lr, momentum=0.9,nesterov=True,weight_decay=wd)

if torch.cuda.is_available():
    vgg = vgg.cuda()
    loss = loss.cuda()

best_val_acc = 0

for epoch in range(num_epochs):
    epoch_st = time.time()
    avg_train_loss = 0
    val_acc = 0
    n = 0

    vgg.train()
    for bidx, data in enumerate(tqdm(train_loader)):
        images, labels = data
        avg_train_loss += train_model(images,labels,vgg,optimizer,loss)

    vgg.eval()
    for bidx, data in enumerate(tqdm(val_loader)):
        images, labels = data
        n += len(labels)
        val_acc += predict_model(images,labels,vgg)

    avg_train_loss /= len(train_loader)
    val_acc /= n

    epoch_et = time.time()
    print(f'Epoch: {epoch+1:02} | Training Loss: {avg_train_loss} | Validation Accuracy: {val_acc} | Time: {epoch_et - epoch_st}s')

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(vgg, "./models/wd{}_best.pth".format(wd))

print("------------ Training Ends --------------")
print("Loading best model...")
net = torch.load("./models/new_wd{}_best.pth".format(wd))
if torch.cuda.is_available():
    net = net.cuda()
    
net.eval()
val_acc = 0
n = 0
for bidx, data in enumerate(tqdm(val_loader)):
    images, labels = data
    n += len(labels)
    val_acc += predict_model(images,labels,net)

val_acc /= n

test_acc = 0
n = 0
for bidx, data in enumerate(tqdm(test_loader)):
    images, labels = data
    n += len(labels)
    test_acc += predict_model(images,labels,net)

test_acc /= n
print(f'Final Validation Accuracy: {val_acc} | Final Test Accuracy: {test_acc}')
