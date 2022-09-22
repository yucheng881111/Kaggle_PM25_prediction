# -*- coding: utf-8 -*-
"""
Created on Sun Sep  4 20:34:00 2022

@author: user
"""

import pandas as pd
from torch.utils.data import Dataset
import numpy as np
import torch

def getData(mode):
    if mode == 'train':
        df = pd.read_csv('train.csv')
        '''
        data = []
        label = []
        
        for i in range(0, len(df)-2, 3):
            if df.iloc[i, 5] != 0 and df.iloc[i+1, 5] != 0 and df.iloc[i+2, 5] != 0:
                label.append([df.iloc[i, 5], df.iloc[i+1, 5], df.iloc[i+2, 5]])
                if df.iloc[i, 6] == 'aribox':
                    data.append([
                        [df.iloc[i, 0], df.iloc[i, 1], df.iloc[i, 2], df.iloc[i, 3], df.iloc[i, 4], 0],
                        [df.iloc[i+1, 0], df.iloc[i+1, 1], df.iloc[i+1, 2], df.iloc[i+1, 3], df.iloc[i+1, 4], 0],
                        [df.iloc[i+2, 0], df.iloc[i+2, 1], df.iloc[i+2, 2], df.iloc[i+2, 3], df.iloc[i+2, 4], 0]
                        ])
                else:
                    data.append([
                        [df.iloc[i, 0], df.iloc[i, 1], df.iloc[i, 2], df.iloc[i, 3], df.iloc[i, 4], 1],
                        [df.iloc[i+1, 0], df.iloc[i+1, 1], df.iloc[i+1, 2], df.iloc[i+1, 3], df.iloc[i+1, 4], 1],
                        [df.iloc[i+2, 0], df.iloc[i+2, 1], df.iloc[i+2, 2], df.iloc[i+2, 3], df.iloc[i+2, 4], 1]
                        ])
        
        return data, label
        '''
        
        data = []
        label = []
        i = 0
        while i < len(df):
            if df.iloc[i, 2] == 0:
                ok = True
                li = []
                li_label = []
                j = 0
                while j < 24:
                    if i+j >= len(df) or df.iloc[i+j ,2] != j:
                        ok = False
                        break
                    
                    li_label.append(df.iloc[i+j, 5])
                    if df.iloc[i+j, 6] == 'aribox':
                        li.append([df.iloc[i+j, 1], df.iloc[i+j, 3], df.iloc[i+j, 4], 0])
                    else:
                        li.append([df.iloc[i+j, 1], df.iloc[i+j, 3], df.iloc[i+j, 4], 1])
                    j += 1
                    
                if ok:
                    data.append(li)
                    label.append(li_label)
                i += j
            else:
                i += 1
                
            
        
        return data, label
    
    else:
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
            
        return target_data, label
                


class PM25Loader(Dataset):
    def __init__(self, mode):
        self.data, self.label = getData(mode)
        print('data preprocessing done.')
        self.mode = mode

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        if self.mode == 'train':
            return np.array(self.data[index]), np.array(self.label[index])
        else:
            return np.array(self.data[index])
    

#data, label = getData('train')




