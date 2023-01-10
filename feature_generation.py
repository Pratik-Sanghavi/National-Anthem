from pyAudioAnalysis import audioBasicIO
from pyAudioAnalysis.ShortTermFeatures import feature_extraction
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import os


def preProcess(file_name):
  (Fs, x) = audioBasicIO.read_audio_file(file_name)
  if( len(x.shape) > 1 and x.shape[1] == 2):
    x = np.mean(x, axis = 1, keepdims = True)
  else:
    x = x.reshape(x.shape[0], 1)
  
  F, f_names = feature_extraction(
    x[:, 0],
    Fs, 0.05*Fs,
    0.025*Fs
  )
  return (f_names, F)
  
def getChromagram(audioData):
  temp_data = audioData[21].reshape(
    1,
    audioData[21].shape[0]
  )
  chronograph = temp_data
  for i in range(22, 33):
    temp_data = audioData[i].reshape(
      1, audioData[i].shape[0]
    )
    chronograph = np.vstack([chronograph, temp_data])
  
  return chronograph

def getNoteFrequency(chromagram):
  numberOfWindows = chromagram.shape[1]
  freqVal = chromagram.argmax(axis = 0)
  histogram, bin = np.histogram(freqVal, bins = 12)
  normalized_hist = histogram.reshape(1, 12).astype(float)/numberOfWindows
  return normalized_hist

DATA_PATH = 'national_anthem_scrape/national_anthem_dataset/audio_files'
OUTFILE = 'national_anthem_scrape/national_anthem_dataset/features.csv'
def getDataset(filePath):
  fileList = os.listdir(filePath)
  X = pd.DataFrame()
  columns = [ "G#", "G", "F#", "F", "E", "D#", "D", "C#", "C", "B", "A#", "A" ]
  for file in tqdm(fileList):
    feature_name, features = preProcess(os.path.join(filePath, file))
    chromagram = getChromagram(features)
    noteFrequency = getNoteFrequency(chromagram)
    x_new = pd.Series(noteFrequency[0, :])
    X = pd.concat([X, x_new], axis = 1)
  data = X.T.copy()
  data.columns = columns
  data.index = [i for i in range(0, data.shape[0])]
  data['file_name'] = fileList
  data = data[list([data.columns[-1]]) + list(data.columns[:-1])]
  return data

data = getDataset(DATA_PATH)
data.to_csv(OUTFILE, index = False)