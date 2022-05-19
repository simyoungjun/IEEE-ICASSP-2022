import librosa
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# %matplotlib inline
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from PIL import Image
import pathlib
import csv
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import keras
from keras import layers
from keras import layers
import keras
import matplotlib.pyplot as plt
import librosa.display

###############영준

volume_path = 'C:/Users/GJ/Desktop/연구실/2022SPCUP/WAV_Ordered train set'

FilePathList =[]

for dirName, subdirList, fileList in os.walk(volume_path):
    for filename in fileList:
        if '.csv' not in filename:
            # print(filename)
            FilePathList.append(volume_path+'/'+filename)
        else:
                    pd_label = pd.read_csv(volume_path+'/'+filename)

labels = np.array(pd_label['algorithm'])

sampling_rate = 16000
# n_fft = 2048 #default



###### Sample mel
X_aug = []
y_labels_aug = []
for i, filepath in enumerate(FilePathList[0:3]):
    # fig, ax = plt.subplots()
    y, sr = librosa.load(filepath, mono=True)
    index_f = 0
    for j in range(y.size//sampling_rate):
        X_aug_seg = y[index_f:index_f+sampling_rate]
        X_aug_seg = np.expand_dims(X_aug_seg, axis=0)
        X_aug.append(X_aug_seg)
        y_labels_aug.append(labels[i])
        index_f = index_f+sampling_rate
X_aug = np.concatenate(X_aug,axis = 0)


    # S = librosa.feature.melspectrogram(y=y, sr=sampling_rate, n_mels=128,
    #                                 fmax=8000)
    # S_dB = librosa.power_to_db(S, ref=np.max)
    # # librosa.display.specshow(S_dB, sr=sr,
    # #                         fmax=8000, ax=ax)
    # #plt.specgram(y, NFFT=2048, Fs=16000, Fc=0, noverlap=128, sides='default', mode='default', scale='dB');
    # # plt.axis('off');
    # break


X_arr = np.zeros((1,128,216))

length_y = []
length_S_dB = []
mfcc_shape = []
for filepath in FilePathList[1:10]:
    # fig, ax = plt.subplots()
    y, sr = librosa.load(filepath, mono=True, duration=5)
    S = librosa.feature.melspectrogram(y=y, sr=sampling_rate, n_mels=128,
                                    fmax=8000)


    S_dB = librosa.power_to_db(S, ref=np.max)
    mfcc = librosa.feature.mfcc(S= S_dB, n_mfcc = 20)

    # print(S_dB.shape)
    # print( y.shape)
    length_y.append(y.shape)
    length_S_dB.append(S_dB.shape)
    mfcc_shape.append(mfcc.shape)
    # X_arr = np.append(X_arr, S_dB,axis=0)

    # librosa.display.specshow(S_dB, sr=sr,
    #                         fmax=8000, ax=ax)

    # plt.show()

length_y = np.array(length_y)
length_S_dB = np.array(length_S_dB)
print(np.unique(length_y))
print(np.unique(length_S_dB))
print('############')





















#
#
# ##### 형섭 :::  wav 파일 스펙토그램 png 파일로 변환 후 저장하기
#
# volume_path = 'C:/Users/GJ/Desktop/연구실/2022SPCUP/WAV_Ordered train set'
#
# FilePathList =[]
#
# for dirName, subdirList, fileList in os.walk(volume_path):
#     for filename in fileList:
#         if '.csv' not in filename:
#             print(filename)
#             FilePathList.append(os.path.join(dirName , filename))
#
#
#
# sampling_rate = 16000
# n_fft = 2048 #default
#
#
#
# ###### Sample mel
# for filename in os.listdir(volume_path):
#     audioname = volume_path+'/'+filename
#     fig, ax = plt.subplots()
#     y, sr = librosa.load(audioname, mono=True, duration=5)
#     S = librosa.feature.melspectrogram(y=y, sr=sampling_rate, n_mels=128,
#                                     fmax=8000)
#     S_dB = librosa.power_to_db(S, ref=np.max)
#     librosa.display.specshow(S_dB, sr=sr,
#                             fmax=8000, ax=ax)
#     #plt.specgram(y, NFFT=2048, Fs=16000, Fc=0, noverlap=128, sides='default', mode='default', scale='dB');
#     plt.axis('off');
#     break
#
#
# X_arr = np.zeros((1,128,216))
#
#
# for filename in os.listdir(volume_path):
#     audioname = volume_path+'/'+filename
#     y, sr = librosa.load(audioname, mono=True, duration=5)
#     S = librosa.feature.melspectrogram(y=y, sr=sampling_rate, n_mels=128,
#                                     fmax=8000)
#     S_dB = librosa.power_to_db(S, ref=np.max)
#
#     X_arr = np.append(X_arr, S_dB)
#
# X_arr = X_arr[1:]
#     # plt.savefig(f'/content/drive/MyDrive/Dataset/mel_spectogram_500/{filename[:-3].replace(".", "")}.png')
#     # plt.clf()