# -*- coding: utf-8 -*-
"""
Created on Sat Mar 21 20:01:09 2020

@author: MSPL
"""
from STFT import STFT
import os
    
def test(wav):
    stft = STFT()
    path = os.path.join('samples', wav)
    stft.load_wav(path)
    stft.transform_matrix()
    stft.plot()
    
test('stereo_sample.wav')