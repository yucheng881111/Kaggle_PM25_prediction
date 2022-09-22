# -*- coding: utf-8 -*-
"""
Created on Tue Sep  6 02:34:35 2022

@author: user
"""

import pandas as pd
from dataloader import PM25Loader
import torch.nn as nn
import torch
import torchvision.models as models
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from models import LSTMTagger


df = pd.read_csv('target.csv')
target_data = []
label = []
for i in range(0, len(df), 3):
    lat = df.iloc[i, 4]
    lon = df.iloc[i, 5]
    li = []
    for _ in range(24):
        li.append([7, lat, lon, 1])
        
    target_data.append(li)
    
test_data = np.array(target_data)
bs = 16
test_loader = torch.utils.data.DataLoader(dataset = test_data, batch_size = bs, shuffle = False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = LSTMTagger(bs, 4, 10)
model.load_state_dict(torch.load('weights.pt'))
model.to(device)
model.eval()

ans = []
for i, data in enumerate(test_loader):
    data = data.to(device, dtype = torch.float)
    outputs = model(data)
    outputs = outputs.view(-1, 24)
    
    
    for tmp in outputs:
        ans.append(float(tmp[-3]))
        ans.append(float(tmp[-2]))
        ans.append(float(tmp[-1]))
        
np.savetxt("pred.csv", ans, delimiter =",")
        

    
        


