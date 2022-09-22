# -*- coding: utf-8 -*-
"""
Created on Sun Sep  4 22:57:10 2022

@author: user
"""

from dataloader import PM25Loader
import torch.nn as nn
import torch
import torch.optim as optim
from models import LSTMTagger
    

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_data = PM25Loader(mode = 'train')
bs = 16
train_loader = torch.utils.data.DataLoader(dataset = train_data, batch_size = bs, shuffle = True)

model = LSTMTagger(bs, 4, 10)
model.to(device)
loss_func = nn.L1Loss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

num_epochs = 50
total_train_accuracy = []
total_test_accuracy = []

model.train()
for epoch in range(num_epochs):
    cnt = 0
    total_loss = 0
    for i, (data, labels) in enumerate(train_loader):
        data = data.to(device, dtype = torch.float)
        labels = labels.to(device, dtype = torch.float)
        
        if labels.size(0) == bs:
            optimizer.zero_grad()
            outputs = model(data)
            
            train_loss = loss_func(outputs.view(bs, -1), labels)
            total_loss += float(train_loss)
            train_loss.backward()
            optimizer.step()
            
    
    #print(labels)
    print(outputs.view(bs, -1))
    print('epoch ' + str(epoch) + ' loss: ' + str(total_loss))
    print()

torch.save(model.state_dict(), 'weights.pt')
    


