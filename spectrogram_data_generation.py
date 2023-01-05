from pyAudioAnalysis import audioBasicIO
from pyAudioAnalysis import audioFeatureExtraction
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import os


def preprocess(file_name):
  (Fs, x) = audioBasicIO.read_audio_file(file_name)
  if( len(x.shape) > 1 and x.shape[1] == 2):
    x = np.mean(x, axis = 1, keepdims = True)
  else:
    x = x.reshape(x.shape[0], 1)
  
  F, f_names = audioFeatureExtraction.stFeatureExtraction(
    x[:, 0],
    Fs, 0.05*Fs,
    0.025*Fs
  )
  return (f_names, F)
  

DATA_PATH = 'national_anthem_scrape/national_anthem_dataset/audio_files'
