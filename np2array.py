# -*- coding: utf-8 -*-
"""
Created on Sun Nov 10 10:33:32 2019

@author: Shiru
"""
import numpy as np

def np2array(foldername, start_idx, end_idx):
    stacked = np.zeros((end_idx-start_idx+1,224,224,3))
    cnt =0
    for i in range(start_idx,end_idx+1):
        print (i)
        stacked[cnt,:,:,:] = np.load(foldername+'\\frame%d.npy'%i)
        cnt+=1
    return stacked


foldername = r'D:\Fall_Dataset\processed_numpy\chute01\cam1.avi'
stacked = np2array(foldername,3,5)
    
        