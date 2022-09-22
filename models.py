# -*- coding: utf-8 -*-
"""
Created on Tue Sep  6 03:06:14 2022

@author: user
"""

from dataloader import PM25Loader
import torch.nn as nn
import torch
import torchvision.models as models
import torch.nn.functional as F
import torch.optim as optim

class LSTMTagger(torch.nn.Module):
    def __init__(self, batch_size, input_dim, hidden_dim):
        super(LSTMTagger,self).__init__()
        self.batch_size = batch_size
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.lstm = nn.LSTM(input_size=self.input_dim, hidden_size=self.hidden_dim, batch_first=True)
        self.fc = nn.Linear(self.hidden_dim, 1)
        

    def forward(self, inputs):
        # 16 * 24 * 4
        out, self.hidden = self.lstm(inputs)
        # 16 * 24 * 10
        tags = self.fc(out)
        # 16 * 24 * 1
        return tags