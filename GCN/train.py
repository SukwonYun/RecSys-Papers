#!/usr/bin/env python
# coding: utf-8

import time
import argparse
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F
import torch.optim as optim

from utils import load_data, accuracy, mkdir_p
from models import GCN


#Training Settings
parser = argparse.ArgumentParser()
parser.add_argument('--no-cuda', action = 'store_true', default = False, help = 'Disables CUDA training.')
parser.add_argument('--fastmode', action = 'store_true', default = False, help = 'Validate during training pass.')
parser.add_argument('--seed', type = int, default = 32, help = 'Random seed.')
parser.add_argument('--epochs', type = int, default = 200, help = 'Number of epochs to train.')
parser.add_argument('--lr', type = float, default = 0.01, help = 'Initial learning rate.')
parser.add_argument('--weight_decay', type = float, default = 5e-4, help = 'L2 loss on parameters.')
parser.add_argument('--hidden', type = int, default = 16, help = 'Number of hidden units.')
parser.add_argument('--dropout', type = float, default = 0.5, help = 'Dropout rate.')


args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()


np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)


# Load data
adj, features, labels, idx_train, idx_val, idx_test = load_data()

# Model and optimizer
model = GCN(n_feat= features.shape[1],
            n_hidden = args.hidden,
            n_class = labels.max().item() + 1,
            dropout = args.dropout)

optimizer = optim.Adam(model.parameters(), lr = args.lr, weight_decay = args.weight_decay)


if args.cuda:
    model.cuda()
    features = features.cuda()
    adj = adj.cuda()
    labels = labels.cuda()
    idx_train = idx_train.cuda()
    idx_val = idx_val.cuda()
    idx_test = idx_test.cuda()

train_losses, train_accs, val_losses, val_accs = [], [], [], []
    
def train(epoch):
    start_time = time.time()
    model.train()
    optimizer.zero_grad()
    output = model(features, adj)
    loss_train = F.nll_loss(output[idx_train], labels[idx_train])
    acc_train = accuracy(output[idx_train], labels[idx_train])
    loss_train.backward()
    optimizer.step()
    
    if not args.fastmode:
        #Evaluate validation set performance separately
        model.eval()
        output = model(features, adj)
    
    loss_val = F.nll_loss(output[idx_val], labels[idx_val])
    acc_val = accuracy(output[idx_val], labels[idx_val])
    
    train_losses.append(loss_train.item())
    train_accs.append(acc_train.item())
    val_losses.append(loss_val.item())
    val_accs.append(acc_val.item())
    
    print('Epoch: {:04d}'.format(epoch+1),
         'loss_train: {:.4f}'.format(loss_train.item()),
          'acc_train: {:.4f}'.format(acc_train.item()),
          'loss_val: {:.4f}'.format(loss_val.item()),
          'acc_val: {:.4f}'.format(acc_val.item()),
          'time: {:.4f}s'.format(time.time()-start_time))


def test():
    model.eval()
    output = model(features, adj)
    loss_test = F.nll_loss(output[idx_test], labels[idx_test])
    acc_test = accuracy(output[idx_test], labels[idx_test])
    print('Test set results:',
         'loss= {:.4f}'.format(loss_test.item()),
         'accuracy= {:.4f}'.format(acc_test.item()))

    return acc_test.data.item()


#Trainin start
time_total = time.time()
for epoch in range(args.epochs):
    train(epoch)
print('Opimization Finished!!')
print('Total time spend: {:.4f}s'.format(time.time()- time_total))

#Test
acc_test = test()

#Plot
output_dir = "results/random_seed_" + str(args.seed)
mkdir_p(output_dir)

fig, ax = plt.subplots()
ax.plot(train_losses, label = 'train loss')
ax.plot(val_losses, label = 'validation loss')
ax.set_xlabel('epochs')
ax.set_ylabel('cross entropy loss')
ax.legend()

ax.set(title="Loss Curve of GCN")
ax.grid()

fig.savefig("results/"+ "random_seed_" + str(args.seed) + "/" + "_loss_curve.png")
plt.close()

fig, ax = plt.subplots()
ax.plot(train_accs, label = 'train accuracy')
ax.plot(val_accs, label = 'validation accuracy')
ax.set_xlabel('epochs')
ax.set_ylabel('accuracy')
ax.legend()

ax.set(title="Accuracy Graph of GCN " + "with Test Accuracy %.4f"%(acc_test))
ax.grid()

fig.savefig("results/"+ "random_seed_" + str(args.seed) + "/" + "_accuracy.png")
plt.close()
