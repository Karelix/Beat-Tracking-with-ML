import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display as dis
import os
import pandas as pd
import json
import math
import random

#def audiogen():
#  l = []
#  for root, dirs, files in os.walk('.'):
#    for filename in files:
#      ext = os.path.splitext(filename)[1]
#      if ext == ".mp3" or ext == ".flac" or ext == ".wav":
#        l.append(os.path.join(root, filename))
#  return l

def load_songs(batch,duration,offset):
  l = []
  for (n,ext) in batch:
    print('track'+str(n)+ext)
    y,sr = librosa.load('tracks/track'+str(n)+ext,duration=duration,offset=offset)
    #tempo, beat_frames = librosa.beat.beat_track(y=y,sr=sr)
    l.append(y)
  return l

#def mel_upsample(mel,rate):
#  mel_up = []
#  mel = mel.transpose()
#  for frame in mel[:,]:
#    for i in range(int(rate)):
#      mel_up.append(frame)
#  mel_up = np.array(mel_up).transpose()
#  return mel_up

#def gen_times(sr,hops,duration,offset):
#  frames_per_sec = math.floor(sr/hops)
#  interval = 1/frames_per_sec
#  curr = offset
#  fin = offset + duration
#  l = []
#  while curr <= fin:
#    l.append(curr)
#    curr += interval
#  return l
     
def gen_ground_truth(time_list,sr,hops,duration,offset):
  frames_per_sec = math.floor(sr/hops)
  print(frames_per_sec)
  interval = 1/frames_per_sec
  print(interval)
  curr = offset
  fin = offset + duration
  ground_truth = []
  for t in time_list:
    if t + interval < offset:
      continue
    elif t - interval > fin and curr > fin:
      break
    else:
      while abs(t-curr) >= interval and curr <= fin:
        ground_truth.append(0)
        curr += interval
      if abs(t-curr) < interval and curr <= fin:
        print(t)
        ground_truth.append(1)
        curr += interval
  return ground_truth

def random_batch(exc):
  nums = []
  batch_list = []
  batch = 32
  for i in range(batch):
    n = random.randint(0,66)
    while n in nums or n in exc:
      n = random.randint(0,66)
    mp3 = os.path.exists('tracks/track'+str(n)+'.mp3')
    flac = os.path.exists('tracks/track'+str(n)+'.flac')
    if mp3:
      batch_list.append((n,'.mp3'))
      nums.append(n)
    elif flac:
      batch_list.append((n,'.flac'))
      nums.append(n)
    else:
      batch_list.append((n,'.m4a'))
      nums.append(n)
  return batch_list
    
def excluded_tracks():
  exc = []
  for i in range(67):
    mp3 = os.path.exists('tracks/track'+str(i)+'.mp3')
    flac = os.path.exists('tracks/track'+str(i)+'.flac')
    m4a = os.path.exists('tracks/track'+str(i)+'.m4a')
    if not mp3 and not m4a and not flac:
      exc.append(i)
  return exc



# Preprocessing

# Tracks I did not find
exc = excluded_tracks()

# Load the dataset
with open('../Datasets/songs.json') as f:
   dataset = json.load(f)

dataset = pd.DataFrame(dataset)
#dataset = dataset.iloc[[i for i in range(67) if i not in exc],:]
# Times of beats in every track 
beats = dataset.iloc[:,2].values

# Windows and Hop length tha I use for STFT and sampling rate for loading the tracks
nft_large = 4096
nft_medium = 2048
nft_small = 512
hops = nft_small/4
sr = 22050

# Building RNN
from keras.models import Sequential
from keras.layers import Dense, TimeDistributed
from keras.layers import LSTM
from keras.layers import Dropout
from keras.layers import Bidirectional

# Initialising RNN
model = Sequential()

# Input Layer
units = 256
model.add(Bidirectional(LSTM(units=units,return_sequences=True),input_shape=))

# Second Layer
model.add(Bidirectional(LSTM(units=units,return_sequences=True)))

# Output Layer
output_units = 517
model.add(Dense(output_units=output_units))

for epoch in range(100):
  # Generate train batch for current epoch
  batch_list = random_batch(exc)
  # Load the waves for the current epoch
  waves = load_songs(batch_list,3,30)
  # Generate ground truths for the current epoch
  ground_truths = [gen_ground_truth(beats[i],22050,hops,3,30) for (i,_) in batch_list]
  
  # Generate training inputs for the current epoch
  mel_large =  librosa.feature.melspectrogram(y=waves[1],sr=sr,n_fft=nft_large,fmin=0,fmax=8000,hop_length=int(nft_small/4))
  mel_medium =  librosa.feature.melspectrogram(y=waves[1],sr=sr,n_fft=nft_medium,fmin=0,fmax=8000,hop_length=int(nft_small/4))
  mel_small =  librosa.feature.melspectrogram(y=waves[1],sr=sr,n_fft=nft_small,fmin=0,fmax=8000,hop_length=int(nft_small/4))
  x_train = mel_large.tolist()
  x_train += mel_medium.tolist()
  x_train += mel_small.tolist()
  x_train = np.array(x_train)
  
  for i in range(len(waves)):
    mel_large =  librosa.feature.melspectrogram(y=waves[i],sr=sr,n_fft=nft_large,fmin=0,fmax=8000,hop_length=int(nft_small/4))
    mel_medium =  librosa.feature.melspectrogram(y=waves[i],sr=sr,n_fft=nft_medium,fmin=0,fmax=8000,hop_length=int(nft_small/4))
    mel_small =  librosa.feature.melspectrogram(y=waves[i],sr=sr,n_fft=nft_small,fmin=0,fmax=8000,hop_length=int(nft_small/4))
    x_train = mel_large.tolist()
    x_train += mel_medium.tolist()
    x_train += mel_small.tolist()
    x_train = np.array(x_train)















































#y, sr = librosa.load('Dystopia.mp3',offset=30.0,duration=30.0)






stft_large = librosa.stft(y,n_fft=nft_large)
mel_large =  librosa.feature.melspectrogram(y=y,sr=sr,n_fft=nft_large,fmin=0,fmax=8000,hop_length=int(nft_small/4))
stft_medium = librosa.stft(y,n_fft=nft_medium)
mel_medium =  librosa.feature.melspectrogram(y=y,sr=sr,n_fft=nft_medium,fmin=0,fmax=8000,hop_length=int(nft_small/4))
stft_small = librosa.stft(y,n_fft=nft_small)
mel_small =  librosa.feature.melspectrogram(y=y,sr=sr,n_fft=nft_small,fmin=0,fmax=8000,hop_length=int(nft_small/4))

# Plotting spectrograms#
plt.figure()
dis.specshow(librosa.amplitude_to_db(stft_large,ref=1.),y_axis='log',x_axis='time')
plt.title('Power Spectrogram (Large N_ft)')
plt.tight_layout()
# Mel spectrogram large window
plt.figure(figsize=(10,4))
dis.specshow(librosa.power_to_db(mel_large,ref=1.),y_axis='mel',fmax=8000,x_axis='frames')
plt.colorbar(format='%+2.0f dB')
plt.title('Mel spectrogram (Large Nfft)')
plt.tight_layout()

plt.figure()
dis.specshow(librosa.amplitude_to_db(stft_medium,ref=1.),y_axis='log',x_axis='time')
plt.title('Power Spectrogram (Medium N_ft)')
plt.tight_layout()
# Mel spectrogram medium window
plt.figure(figsize=(10,4))
dis.specshow(librosa.power_to_db(mel_medium,ref=1.),y_axis='mel',fmax=8000,x_axis='frames')
plt.colorbar(format='%+2.0f dB')
plt.title('Mel spectrogram (Medium Nfft)')
plt.tight_layout()

plt.figure()
dis.specshow(librosa.amplitude_to_db(stft_small,ref=1.),y_axis='log',x_axis='time')
plt.title('Power Spectrogram (Small N_ft)')
plt.tight_layout()
# Mel spectrogram small window
plt.figure(figsize=(10,4))
dis.specshow(librosa.power_to_db(mel_small,ref=1.),y_axis='mel',fmax=8000,x_axis='frames')
plt.colorbar(format='%+2.0f dB')
plt.title('Mel spectrogram (Small Nfft)')
plt.tight_layout()

mel_large = librosa.power_to_db(mel_large,ref=1.)
mel_medium = librosa.power_to_db(mel_medium,ref=1.)
mel_small = librosa.power_to_db(mel_small,ref=1.)

# Plotting after upsampling
# Mel spectrogram large window
plt.figure(figsize=(10,4))
dis.specshow(librosa.power_to_db(mel_large,ref=1.),y_axis='mel',fmax=8000,x_axis='frames')
plt.colorbar(format='%+2.0f dB')
plt.title('Mel spectrogram (Large Nfft)')
plt.tight_layout()

# Mel spectrogram medium window
plt.figure(figsize=(10,4))
dis.specshow(librosa.power_to_db(mel_medium,ref=1.),y_axis='mel',fmax=8000,x_axis='frames')
plt.colorbar(format='%+2.0f dB')
plt.title('Mel spectrogram (Medium Nfft)')
plt.tight_layout()

# Mel spectrogram small window
plt.figure(figsize=(10,4))
dis.specshow(librosa.power_to_db(mel_small,ref=1.),y_axis='mel',fmax=8000,x_axis='frames')
plt.colorbar(format='%+2.0f dB')
plt.title('Mel spectrogram (Small Nfft)')
plt.tight_layout()

