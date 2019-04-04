import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display as dis
import os

def audiogen():
  l = []
  for root, dirs, files in os.walk('.'):
    for filename in files:
      ext = os.path.splitext(filename)[1]
      if ext == ".mp3" or ext == ".flac" or ext == ".wav":
        l.append(os.path.join(root, filename))
  return l

def load_songs(songs,duration):
  l = []
  for song in songs:
    print(song)
    y,sr = librosa.load(song,duration=duration)
    #tempo, beat_frames = librosa.beat.beat_track(y=y,sr=sr)
    l.append(y)
  return l

def mel_upsample(mel,rate):
  mel_up = []
  mel = mel.transpose()
  for frame in mel[:,]:
    for i in range(int(rate)):
      mel_up.append(frame)
  mel_up = np.array(mel_up).transpose()
  return mel_up

y, sr = librosa.load('Dystopia.mp3', duration=30.0,offset=30.0)
#plt.figure()
#dis.waveplot(y,sr=sr)
nft_large = 4096
nft_medium = 2048
nft_small = 512

stft_large = librosa.stft(y,n_fft=nft_large)
mel_large =  librosa.feature.melspectrogram(y=y,sr=sr,n_fft=nft_large,fmin=0,fmax=8000,hop_length=int(nft_large/4))
stft_medium = librosa.stft(y,n_fft=nft_medium)
mel_medium =  librosa.feature.melspectrogram(y=y,sr=sr,n_fft=nft_medium,fmin=0,fmax=8000,hop_length=int(nft_medium/4))
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

mel_large = mel_upsample(mel_large,len(mel_small[0,:])/len(mel_large[0,:]))
mel_medium = mel_upsample(mel_medium,len(mel_small[0,:])/len(mel_medium[0,:]))

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

