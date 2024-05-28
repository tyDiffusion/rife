#!/bin/env python

import _thread
import os
import time
import shutil
import tempfile
import warnings
import subprocess
from queue import Queue
#import filetype
import cv2
import numpy as np
import torch
from torch.nn import functional as F
#from rich import print # pylint: disable=redefined-builtin
#from rich.pretty import install as install_pretty
#from rich.traceback import install as install_traceback
from tqdm import tqdm

from model.ssim import ssim_matlab
from model.RIFE_HDv3 import Model


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def getRIFEModel(modelPath, gpu, half):
    torch.set_grad_enabled(False)
    if gpu and torch.cuda.is_available():
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True
        if half:
            torch.set_default_tensor_type(torch.cuda.HalfTensor)
    model = Model()
    model.load_model(modelPath, -1)
    model.eval()
    model.device()
    
    return model
    

def interpTwoFramesRIFE(model, gpu, half, img1, img2, inter_frames, save_num, save_name, save_path):
        
    count = 0
    scale = 1
    
    newFrames = []
    buffer = Queue(maxsize=8192)
    
    if (model  is None):
        print ("\nRIFE: error - model not loaded")
        return
    
    imgs = [img1, img2]

    frame = cv2.imread(imgs[0], cv2.IMREAD_UNCHANGED)[:, :, ::-1].copy()
    h, w, _ = frame.shape
    
    def write():       
        nonlocal save_num
        nonlocal newFrames
        nonlocal buffer
        while buffer.qsize() > 0:
            item = buffer.get()
            filename = save_path + save_name + str(save_num) + ".png"
            save_num += 1      
            newFrames.append(filename)
            cv2.imwrite(filename, item[:, :, ::-1])

    def execute(I0, I1, n):
        if model.version >= 3.9:
            res = []
            for i in range(n):
                interp = (i+1) * 1. / (n+1)
                res.append(model.inference(I0, I1, interp, scale))
            return res
        else:
            middle = model.inference(I0, I1, scale)
            if n == 1:
                return [middle]
            first_half = execute(I0, middle, n=n//2)
            second_half = execute(middle, I1, n=n//2)
            if n % 2:
                return [*first_half, middle, *second_half]
            else:
                return [*first_half, *second_half]
                

    def pad(img):
        return F.pad(img, padding).half() if half else F.pad(img, padding) # pylint: disable=not-callable

    tmp = max(128, int(128 / scale))
    ph = ((h - 1) // tmp + 1) * tmp
    pw = ((w - 1) // tmp + 1) * tmp
    padding = (0, pw - w, 0, ph - h)
        
    buffer.put(frame)   
    
    I1 = pad(torch.from_numpy(np.transpose(frame, (2,0,1))).to(device, non_blocking=True).unsqueeze(0).float() / 255.)    
    for f in range(1, len(imgs)):
        frame = cv2.imread(imgs[f], cv2.IMREAD_UNCHANGED)[:, :, ::-1].copy()
        I0 = I1
        I1 = pad(torch.from_numpy(np.transpose(frame, (2,0,1))).to(device, non_blocking=True).unsqueeze(0).float() / 255.)
        
        output = execute(I0, I1, inter_frames) 
        for mid in output:
            mid = (((mid[0] * 255.).byte().cpu().numpy().transpose(1, 2, 0)))
            buffer.put(mid[:h, :w]) 
        buffer.put(frame)
            
            
    write()

    while not buffer.empty(): #wait for buffer to empty while files finish saving...        
        time.sleep(0.01)
        
        
    return newFrames
