#!/usr/bin/env python
# coding: utf-8

# In[11]:


import numpy as np
import pandas as pd
import torch
import tqdm
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader
from pathlib import Path

# In[19]:

from fm import FM


# In[7]:


class MovieLens1M(torch.utils.data.Dataset):
    """
    MovieLens 1M Dataset
    Treat as negative samples when rating is less than 3
    """
    
    def __init__(self, dataset_path, sep='::', engine = 'python', header = None):
        data = pd.read_csv(dataset_path, sep=sep, engine=engine, header=header).to_numpy()[:,:3]
        self.items = data[:, :2].astype(np.int) - 1 #since index starts from 0
        self.targets = self.__preprocess_target(data[:,2]).astype(np.float32)
        self.field_dims = np.max(self.items, axis = 0) + 1
        self.user_field_idx = np.array((0, ), dtype = np.long)
        self.item_field_idx = np.array((1, ), dtype = np.long)
        
    def __len__(self):
        return self.targets.shape[0]
    
    def __getitem__(self, index):
        return self.items[index], self.targets[index]
    
    def __preprocess_target(self, target):
        target[target <= 3] = 0
        target[target > 3] = 1
        
        return target


# In[20]:


def get_dataset(name, path):
    if name == 'movielens1M':
        return MovieLens1M(path)
    else:
        raise ValueError('unknown dataset name: ' + name)

def get_model(name,dataset):
    """
    Hyperparameters are empirically determined
    """
    
    field_dims = dataset.field_dims
    
    if name == 'fm':
        return FM(field_dims, embed_dim=16)
    else:
        raise ValueError('unknown dataset name: ' + name)


# In[21]:


class EarlyStopper(object):
    
    def __init__(self, num_trials, save_path):
        self.num_trials = num_trials
        self.trial_counter = 0
        self.best_accuracy = 0
        self.save_path = save_path
    
    def is_continuable(self, model, accuracy):
        if accuracy > self.best_accuracy:
            self.best_accuracy = accuracy
            self.trial_counter = 0
            torch.save(model, self.save_path)
            
            return True
        
        elif self.trial_counter + 1 < self.num_trials:
            self.trial_counter += 1
            
            return True
        
        else:
            return False
        


# In[24]:


def train(model, optimizer, data_loader, criterion, device, log_interval = 100):
    model.train()
    total_loss = 0
    tk0 = tqdm.tqdm(data_loader, smoothing = 0, mininterval = 1.0)
    for i, (fields, target) in enumerate(tk0):
        fields, target = fields.to(device), target.to(device)
        y = model(fields)
        loss = criterion(y, target.float())
        model.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        if (i+1) % log_interval == 0:
            tk0.set_postfix(loss = total_loss / log_interval)
            total_loss = 0
            
def test(model, data_loader, device):
    model.eval()
    targets, predicts = list(), list()
    with  torch.no_grad():
        for fields, target in tqdm.tqdm(data_loader, smoothing = 0, mininterval = 1.0):
            fields, target = fields.to(device), target.to(device)
            y = model(fields)
            targets.extend(target.tolist())
            predicts.extend(y.tolist())
        
        return roc_auc_score(targets, predicts)
    


# In[25]:


def main(dataset_name,
        dataset_path,
        model_name,
        epoch,
        learning_rate,
        batch_size,
        weight_decay,
        device,
        save_dir):
    device = torch.device(device)
    dataset = get_dataset(dataset_name, Path('.')/'ml-1m'/'ratings.dat')
    train_length = int(len(dataset) * 0.8)
    valid_length = int(len(dataset) * 0.1)
    test_length = len(dataset) - train_length - valid_length
    train_dataset, valid_dataset, test_dataset = torch.utils.data.random_split(
        dataset, (train_length, valid_length, test_length))
    
    train_data_loader = DataLoader(train_dataset, batch_size = batch_size, num_workers=8)
    valid_data_loader = DataLoader(valid_dataset, batch_size = batch_size, num_workers=8)
    test_data_loader = DataLoader(test_dataset, batch_size = batch_size, num_workers=8)
    
    model = get_model(model_name, dataset).to(device)
    criterion = torch.nn.BCELoss()
    optimizer = torch.optim.Adam(params=model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    early_stopper = EarlyStopper(num_trials=2, save_path=f'{save_dir}.pt')
    
    for epoch_i in range(epoch):
        train(model, optimizer, train_data_loader, criterion, device)
        auc = test(model, valid_data_loader, device)
        print('epoch:', epoch_i, 'auc', auc)
        if not early_stopper.is_continuable(model,auc):
            print(f'validation best auc: {early_stopper.best_accuracy}')
            break
        
        auc = test(model, test_data_loader, device)
        print(f'test auc: {auc}')
    


# In[27]:


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', default='movielens1M')
    parser.add_argument('--dataset_path', help='/ml-1m/ratings.dat')
    parser.add_argument('--model_name', default='fm')
    parser.add_argument('--epoch', type=int, default=100)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--batch_size', type=int, default=2048)
    parser.add_argument('--weight_decay', type=float, default=1e-6)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--save_dir', default='FM-checkpoint')
    args = parser.parse_args()
    main(args.dataset_name,
         args.dataset_path,
         args.model_name,
         args.epoch,
         args.learning_rate,
         args.batch_size,
         args.weight_decay,
         args.device,
         args.save_dir)


# In[ ]:




