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
from torch import nn

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

    def test(self):
      PATH = f'model/pretrained/best_{self.config["experiment_desc"]}.pth'
      self.model.load_state_dict(torch.load(PATH)['model'])
      self.model.eval()
      self._validate(-1, True)

    def train(self):

        self._init_params()
        self.model.cuda()
        
        for epoch in range(0, self.epochs):
            self._run_epoch(epoch)
            self._validate(epoch)

            if self.metric_counter.update_best_model():
                torch.save({
                    'model': self.model.state_dict(),
                    'optimizer': self.optimizer.state_dict(),
                    'dataset_params': self.dataset.get_params()
                }, f'model/pretrained/best_{self.config["experiment_desc"]}.pth')

            print(self.metric_counter.loss_message())
            logging.debug(
                f"Experiment Name: {self.config['experiment_desc']}, Epoch: {epoch}, Loss: {self.metric_counter.loss_message()}")


# def train(encoder, decoder, data, epochs=10, batch_size=30, seq_length=500, hidden_size=1000, lr=0.001, clip=5, val_frac=0.1, print_every=10):
    
#     counter = 0

#     for e in range(epochs):
#         # initialize hidden state
#         h = None

#         for lenx, x, leny, y in get_batches(data, batch_size=batch_size, seq_length=seq_length):
#           try:
#             counter += 1
#                         # One-hot encode our data and make them Torch tensors
#             lenx, leny = lenx.cuda(), leny.cuda()
#             inputs, targets = x.cuda(), y.cuda()

#             # Creating new variables for the hidden state, otherwise
#             # we'd backprop through the entire training history
#             h = None

#             # zero accumulated gradients
#             encoder.zero_grad()
#             decoder.zero_grad()
            
#             max_target_length = int(max(leny).item())
#             decoder_input = targets[:, :1]

#             all_decoder_outputs = torch.zeros(batch_size, max_target_length, decoder.output_size)

#             decoder_input = decoder_input.cuda()
#             all_decoder_outputs = all_decoder_outputs.cuda()

#             # Run through decoder one time step at a time
#             encoder_outputs, encoder_hidden = encoder(inputs, lenx, h)

#             decoder_hidden = encoder_hidden[:decoder.n_layers].cuda()

#             for t in range(max_target_length):
#                 decoder_output, decoder_hidden, decoder_attn = decoder(decoder_input, decoder_hidden, encoder_outputs)

#                 all_decoder_outputs[:, t] = decoder_output
#                 decoder_input = targets[:, t+1].unsqueeze(1) # Next input is current target
#             # get the output from the model
#             # h_dec = encoder_outputs
#             # print("Encoder_hidden: ", encoder_hidden.size())
#             # out, h_dec = decoder(targets[:, :-1], encoder_hidden[:decoder.n_layers], h_dec)
#             out = all_decoder_outputs
#             targets = targets[:, :out.size(1)]
#             tar = targets.argmax(2)
#             tar = tar.view(tar.size(0)*tar.size(1))
#             cur = out.view(out.size(0)*out.size(1), -1)




#             loss = criterion(cur, tar)
#             training_loss.append(loss.item())


#             # calculate the loss and perform backprop
#             # loss = criterion(output, targets.view(batch_size*seq_length).long())
#             loss.backward()
#             nn.utils.clip_grad_norm_(decoder.parameters(), clip)
#             nn.utils.clip_grad_norm_(encoder.parameters(), clip)
#             opt2.step()
#             opt1.step()
            
            
#             # loss stats
#             if counter % print_every == 0:
#                     # Get validation loss
#                 val_h = None
#                 val_losses = []
#                 val_acc = []
#                 encoder.eval()
#                 decoder.eval()

#                 for lenx, x, leny, y in get_batches(val_data, batch_size=batch_size, seq_length=seq_length):

#                         # One-hot encode our data and make them Torch tensors
                        
#                         # Creating new variables for the hidden state, otherwise
#                         # we'd backprop through the entire training history
#                     val_h = None
                        
#                     inputs, targets = x, y
#                     inputs, targets = inputs.cuda(), targets.cuda()
#                     lenx, leny = lenx.cuda(), leny.cuda()
                        

#                     encoder_outputs, encoder_hidden = encoder(inputs, lenx, val_h)

#                     max_target_length = int(max(leny).item())
#                     decoder_input = targets[:, :1]

#                     all_decoder_outputs = torch.zeros(batch_size, max_target_length, decoder.output_size)

#                     decoder_input = decoder_input.cuda()
#                     all_decoder_outputs = all_decoder_outputs.cuda()

#                     # Run through decoder one time step at a time
#                     encoder_outputs, encoder_hidden = encoder(inputs, lenx, h)

#                     decoder_hidden = encoder_hidden[:decoder.n_layers].cuda()

#                     for t in range(max_target_length - 1):
#                         decoder_output, decoder_hidden, decoder_attn = decoder(decoder_input, decoder_hidden, encoder_outputs)

#                         all_decoder_outputs[:, t] = decoder_output
#                         decoder_input = targets[:, t+1].unsqueeze(1) # Next input is current target
                    
#                     # get the output from the model
#                     # h_dec = encoder_outputs
#                     # print("Encoder_hidden: ", encoder_hidden.size())
#                     # out, h_dec = decoder(targets[:, :-1], encoder_hidden[:decoder.n_layers], h_dec)
#                     out = all_decoder_outputs
#                     targets = targets[:, :out.size(1)]
#                     # h_dec = encoder_outputs

#                     # out, h_dec = decoder(targets[:, :-1], encoder_hidden[:decoder.n_layers], h_dec)
#                     tar = targets.argmax(2)
#                     tar = tar.view(tar.size(0)*tar.size(1))
#                     cur = out.view(out.size(0)*out.size(1), -1)
#                     val_loss = criterion(cur, tar)
              
#                     validation_loss.append(val_loss.item())
#                     val_losses.append(val_loss.item())
#                     output = cur
#                     ind = min(len(validation_loss), 10)
#                     acc_output = output.cpu().detach()

#                     Y = tar.cpu().view(-1)

#                     Y_hat = acc_output.argmax(1)
#                     Y_hat = Y_hat.view(-1)

#                     tag_pad_token = 1000
#                     mask = (Y < tag_pad_token).float()

#                     nb_tokens = int(torch.sum(mask).item())

#                     current_accuracy =  (np.equal(Y_hat.numpy().astype("int32"), Y.numpy()) * mask.numpy()).sum()

#                     div = nb_tokens
#                     val_acc.append(current_accuracy/(div))
#                     validation_accuracy.append(current_accuracy/(div))
    def _run_epoch_seq2seq(self, epoch):
        self.metric_counter.clear()
        h = None
        counter = 0 
        for lenX, X, leny, y in self.loader_train:
            lenX, perm_idx = torch.from_numpy(lenX).sort(0, descending=True)
            X = X[perm_idx]
            y = y[perm_idx]
            leny = leny[perm_idx]

            X, y = X.cuda(), y.cuda()
            lenX, leny = lenX.cuda(), leny.cuda()

            self.optimizer.zero_grad()

            


    def _run_epoch(self, epoch):
        
        self.metric_counter.clear()
        h = None
        counter = 0 
        for X, y in self.loader_train:
            X, y = X.cuda(), y.cuda()

            self.optimizer.zero_grad()
            output, h = self.model(X, h)
            
            loss = self.loss_fn(output, y.view(y.nelement()).long())
            loss.backward()
            

            h = tuple([i.detach_() for i in h])
            acc = self.calc_acc(output, y.view(y.nelement()))

            nn.utils.clip_grad_norm_(self.model.parameters(), self.config['clip'])
            self.optimizer.step()

            self.metric_counter.add_losses(loss)
            self.metric_counter.add_acc(acc)
            if not counter % self.config['print_every']:
              print(f"Epoch: {epoch}; Train Step: {counter}: {self.metric_counter.loss_message()}")
          
          
            counter += 1

        
        self.metric_counter.write_to_tensorboard(epoch)
        
        
    
            
    def _validate(self, epoch, test=False):
        h = None
        self.model.eval()
        self.metric_counter.clear()
        counter = 0
        loader = (self.loader_test if test else self.loader_val)

        for X, y in loader:
            X, y = X.cuda(), y.cuda()
            output, h = self.model(X, h)

            loss = self.loss_fn(output, y.view(y.nelement()).long())
            acc = self.calc_acc(output, y.view(y.nelement()))

            self.metric_counter.add_losses(loss)
            self.metric_counter.add_acc(acc)
            if not counter % self.config['print_every']:
              print(f"Epoch: {epoch}; Valid Step: {counter}: {self.metric_counter.loss_message()}")
            counter += 1
        
        if not test:
          self.metric_counter.write_to_tensorboard(epoch, validation=True)
        else:
          print(self.metric_counter.loss_message())
        
        self.model.train()
    
    def calc_acc(self, y_pred, y):
        acc_output = y_pred.cpu().detach().numpy().astype("int32")
        acc =  np.equal(acc_output.argmax(1), y.cpu().numpy())
        return acc.sum()/acc_output.shape[0]

  
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

        self.loader_test = data.DataLoader(self.test_dataset,
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
    trainer.test()