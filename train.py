# from __future__ import print_function
import torch
import torch.optim as optim
from torch.utils import data
import numpy as np
import yaml
import logging
from model.model_utils import get_model, get_loss
from dataload.dataloader import DataLoader
from metric_counter import MetricCounter
import os


class Trainer(object):
    def __init__(self, config):
        self.config = config
        self.dataset = self._get_dataset(config["dataroot"], config["seq_length"])
        dlen = len(self.dataset)
        splitlen = [int(0.8*dlen), int(0.1*dlen), 0]
        splitlen[2] = dlen - sum(splitlen)
        
        self.train_dataset, self.val_dataset, self.test_dataset = data.random_split(self.dataset, splitlen)

        self.experiment_name = f"{config['experiment_desc']}_{config['model']['model_n']}"
        self.metric_counter = MetricCounter(self.experiment_name, self.config["print_every"])

    def train(self):
        self._init_params()
        for epoch in range(0, self.epochs):
            self._run_epoch(epoch)
            self._validate(epoch)

            if self.metric_counter.update_best_model():
                torch.save({
                    'model': self.model.state_dict(),
                    'optimizer': self.optimizer.state_dict(),
                    'dataset_params': self.dataset.get_params()
                }, f'best_{self.config["experiment_desc"]}.pth')

            print(self.metric_counter.loss_message())
            logging.debug(
                f"Experiment Name: {self.config['experiment_desc']}, Epoch: {epoch}, Loss: {self.metric_counter.loss_message()}")

    def _run_epoch(self, epoch):

        self.metric_counter.clear()
        h = None
        counter = 0 
        for X, y in self.loader_train:
            X, y = X.cuda(), y.cuda()

            self.optimizer.zero_grad()
            output, h = self.model(X, h)
            loss = self.loss_fn(output, y.view(self.config['batch_size']*self.config['seq_length']).long())
            
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), self.config['clip'])
            self.optimizer.step()

            self.metric_counter.add_losses(loss)

            if counter % self.config['print_every']:
              print(f"Epoch: {epoch}; Train Step: {counter}: {metric_counter.loss_message()}")

          
          
            counter += 1

        
        self.metric_counter.write_to_tensorboard(epoch)
        
        
    
            
    def _validate(self, epoch):
        self.model.eval()
        self.metric_counter.clear()
        counter = 0

        for X, y in self.loader_val:
            X, y = X.cuda(), y.cuda()
            output, h = self.model(X, h)

            loss = self.loss_fn(output, y.view(self.config['batch_size']*self.config['seq_length']).long())

            self.metric_counter.add_losses(loss)
            if counter % self.config['print_every']:
              print(f"Epoch: {epoch}; Valid Step: {counter}: {metric_counter.loss_message()}")
            counter += 1

        self.metric_counter.write_to_tensorboard(epoch, validation=True)
        self.model.train()

  
    def _get_dataset(self, dataroot, seq_length):
        return DataLoader(dataroot, seq_length)

    def _get_optim(self, params):
        if self.config['optimizer']['name'] == 'adam':
            optimizer = optim.Adam(params, lr=self.config['optimizer']['lr'])
        elif self.config['optimizer']['name'] == 'sgd':
            optimizer = optim.SGD(params, lr=self.config['optimizer']['lr'])
        else:
            raise ValueError(f"Optimizer {self.config['optimizer']['name']} not recognized.")
        return optimizer
      
    def _init_params(self):

        self.loader_train = data.DataLoader(self.train_dataset,
                                            batch_size=self.config['batch_size'],
                                            shuffle=True)
        self.loader_val = data.DataLoader(self.val_dataset,
                                          batch_size=self.config['batch_size'],
                                          shuffle=False)
                                          
        self.model = get_model(self.config['model']['model_n'],
                               self.dataset.get_params(),
                               self.config['model']['n_hidden'],
                               self.config['model']['n_layers'],
                               self.config['model']['drop_prob'],
                               self.config['model']['grad_clip'])

        self.epochs = self.config['num_epochs']
        self.optimizer = self._get_optim(self.model.parameters())
        self.loss_fn = get_loss(self.config['model']['loss'])

        


if __name__ == '__main__':
    with open('config/params.yaml', 'r') as f:
        config = yaml.load(f)

    trainer = Trainer(config)
    trainer.train()