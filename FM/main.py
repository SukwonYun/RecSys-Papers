#!/usr/bin/env python
# coding: utf-8


import numpy as np
import pandas as pd
import torch
import tqdm
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader
from pathlib import Path

from fm import FM


class MovieLens1M(torch.utils.data.Dataset):
    """
    MovieLens 1M Dataset
    Treat negative samples when rating is less than 3
    """
    
    def __init__(self, dataset_path, sep='::', engine = 'python', header = None):
        data = pd.read_csv(dataset_path, sep=sep, engine=engine, header=header).to_numpy()[:,:3]
        self.items = data[:, :2].astype(np.int) - 1 #since index starts from 0
        self.targets = self.__preprocess_target(data[:,2]).astype(np.float32)
        self.field_dims = np.max(self.items, axis = 0) + 1
        #self.user_field_idx = np.array((0, ), dtype = np.long)
        #self.item_field_idx = np.array((1, ), dtype = np.long)  
    
    def __preprocess_target(self, target):
        target[target <= 3] = 0
        target[target > 3] = 1
        
        return target
    
    def __len__(self):
        """
        Overwriting
        """
        
        return self.targets.shape[0]
    
    def __getitem__(self, index):
        """
        Overwriting
        """
        
        return self.items[index], self.targets[index]

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
    

def main(dataset_name,
        epoch,
        learning_rate,
        batch_size,
        weight_decay,
        device,
        save_dir):
    
    device = torch.device(device)
    
    dataset = MovieLens1M(Path('.')/'ml-1m'/'ratings.dat')
    field_dims = dataset.field_dims

    
    train_length = int(len(dataset) * 0.8)
    valid_length = int(len(dataset) * 0.1)
    test_length = len(dataset) - train_length - valid_length
    
    #randomly split train/valid/test sets based on their lengths
    train_dataset, valid_dataset, test_dataset = torch.utils.data.random_split(
        dataset, (train_length, valid_length, test_length))
    
    #DataLoader enables new combination of batches in each epoch
    train_data_loader = DataLoader(train_dataset, batch_size = batch_size)
    valid_data_loader = DataLoader(valid_dataset, batch_size = batch_size)
    test_data_loader = DataLoader(test_dataset, batch_size = batch_size)
    
    model = FM(field_dims, embed_dim=16).to(device)
    
    criterion = torch.nn.BCELoss()
    optimizer = torch.optim.Adam(params=model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    early_stopper = EarlyStopper(num_trials=2, save_path=f'{save_dir}.pt')
    
    for i in range(epoch):
        train(model, optimizer, train_data_loader, criterion, device)
        auc = test(model, valid_data_loader, device)
        print('epoch:', i, 'auc', auc)
        
        if not early_stopper.is_continuable(model,auc):
            print(f'validation best auc: {early_stopper.best_accuracy}')
            break
        
        auc = test(model, test_data_loader, device)
        print(f'test auc: {auc}')
    

if __name__ == '__main__':
    
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', default='movielens1M')
    parser.add_argument('--epoch', type=int, default=5)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--batch_size', type=int, default=2048)
    parser.add_argument('--weight_decay', type=float, default=1e-6)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--save_dir', default='FM-checkpoint')
    args = parser.parse_args()
    
    main(args.dataset_name,
         args.epoch,
         args.learning_rate,
         args.batch_size,
         args.weight_decay,
         args.device,
         args.save_dir)




