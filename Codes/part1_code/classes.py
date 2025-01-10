from tensorflow import keras
import matplotlib.pyplot as plt
import os
from scipy.io import wavfile
from collections import defaultdict, Counter
from scipy import signal
import numpy as np
import librosa
import random as rn
from tensorflow.keras.layers import Dense
from tensorflow.keras import Input
# from keras.engine import Model
from tensorflow.keras.layers import Dense, TimeDistributed, Dropout, Bidirectional, GRU, BatchNormalization, Activation, \
    LeakyReLU, LSTM, Flatten, RepeatVector, Permute, Multiply, Conv2D, MaxPooling2D
import librosa
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import os
from PIL import Image
import pathlib
import csv
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from tensorflow.keras import layers
import pandas as pd
from tensorflow.keras.utils import to_categorical
import statsmodels.api as sm
import time
from sklearn.preprocessing import StandardScaler,MinMaxScaler

class data:
    file_split_num = []

    def __init__(self, FilePathList, labels, sampling_rate=16000, n_mels=32, known = True):
        self.FilePathList = FilePathList
        self.labels = labels
        self.known = known
        self.n_mels = n_mels
        self.extract_mel(sampling_rate, n_mels)


    def extract_mel(self, sampling_rate, n_mels):
        X_mel_aug = []
        y_labels_aug = []
        self.file_split_num = []
        for i, filepath in enumerate(self.FilePathList):
            # fig, ax = plt.subplots()
            y, sr = librosa.load(filepath, mono=True, sr=sampling_rate)
            index_f = 0
            for j in range(y.size // sampling_rate):
                X_aug_seg = y[index_f:index_f + sampling_rate]

                S = librosa.feature.melspectrogram(y=X_aug_seg, sr=sampling_rate, n_mels=n_mels, fmax=8000)
                S_dB = S
                # S_dB = librosa.power_to_db(S, ref=np.max)
                X_mel_aug.append(S_dB)

                y_labels_aug.append(self.labels[i])

                index_f = index_f + sampling_rate
            self.file_split_num.append(j + 1)
            #         X_aug_seg = np.expand_dims(X_aug_seg, axis=0)
            #         X_aug.append(X_aug_seg)
        # X_aug = np.concatenate(X_aug,axis = 0)
        X_split = np.array(X_mel_aug)
        y_split = np.array(y_labels_aug)
        #         print('train set.shape : ',X_split.shape)
        #         print('y_train.shape',y_split.shape )
        self.X_split = X_split
        self.y_split = y_split

        if self.known == True:
            self.step1_split_labels = np.zeros(self.y_split.size)
            self.step1_labels = np.zeros(self.labels.size)
        else:
            self.step1_split_labels = np.ones(self.y_split.size)
            self.step1_labels = np.ones(self.labels.size)


    def reshape_data(self):
        self.X_reshaped = self.X_split.reshape(self.X_split.shape[0], self.X_split.shape[1], self.X_split.shape[2], 1)
        print('X_train.shape : ', self.X_reshaped.shape)

        self.y_reshaped = to_categorical(self.y_split)
        # print(y_test.shape)
        print('y_train.shape', self.y_reshaped.shape)

    def flatten(self):
        self.X_flattened = self.X_split.reshape(self.X_split.shape[0], -1)
        print('X_split to X_flattend.shape : ', self.X_flattened.shape)


    def std_scale(self):
        scaler = StandardScaler()
        self.X_std_scaled = scaler.fit_transform(self.X_flattened)
        return self.X_std_scaled

    def min_max_scale(self):
        scaler = MinMaxScaler()
        self.X_min_max_scaled = scaler.fit_transform(self.X_flattened)
        return self.X_min_max_scaled
#         return self.X_reshaped, self.y_reshaped


def std_scale(data):
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)
    return data_scaled

def min_max_scale(data):
    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(data)
    return data
#         return self.X_reshaped, self.y_reshaped


def file_path_list(volume_path):
    FilePathList = []

    for dirName, subdirList, fileList in os.walk(volume_path):
        for filename in fileList:
            if '.csv' not in filename:
                #             print(filename)
                FilePathList.append(volume_path + '/' + filename)
            else:
                pd_label = pd.read_csv(volume_path + '/' + filename)

    labels = np.array(pd_label['algorithm'])
    return FilePathList, labels


class mkde:
    def __init__(self, data, n_mels=32, pa=1):
        self.data = data
        self.n_mels = n_mels
        self.data_reshape()
        self.pa = pa

    def data_reshape(self):
        self.reshaped_data = self.data.transpose((0, 2, 1))
        print(self.reshaped_data.shape)

        self.reshaped_data = self.reshaped_data.reshape((-1, self.reshaped_data.shape[2]))
        print(self.reshaped_data.shape)
        scaler = StandardScaler()
        self.reshaped_data = scaler.fit_transform(self.reshaped_data)
        # mm_scaler = MinMaxScaler()
        # self.reshaped_data = mm_scaler.fit_transform(self.reshaped_data)


    def make_pdf(self):
        var_type = 'c' * self.n_mels

        std_feature = np.std(self.reshaped_data, axis=0)
        print(std_feature.shape)
        d = self.reshaped_data.shape[1]
        feature_length = self.reshaped_data.shape[0]
        c = (4 / (d + 2) / feature_length) ** (1 / (d + 4))
        bw = std_feature * c
        # bw = bw.transpose()

        self.bw = bw * self.pa

        self.dens = sm.nonparametric.KDEMultivariate(
            data=self.reshaped_data, var_type=var_type, bw=bw * self.pa)


def mkde_test(train_file_num, test_file_num, train, data, n_mels=32, pa=1):
    train_file_length = np.sum(train.file_split_num[:train_file_num])

    density = mkde(train.X_split[:train_file_length], n_mels, pa)
    density.make_pdf()
    # print(density.dens)
    # print(density.bw)

    all_time = time.time()
    start = time.time()  # 시작 시간 저장

    unknown_pd = []
    test_file_length = np.sum(data.file_split_num[:test_file_num])
    unknown_d = mkde(data.X_split[:test_file_length], n_mels)
    file_pd = density.dens.pdf(unknown_d.reshaped_data)
    #     print(file_pd)
    unknown_pd.append(file_pd)

    print(len(unknown_pd))
    print("time :", time.time() - start)
    print('all time : ', time.time() - all_time)
    return unknown_pd

def mkde_file_acc(pd, data, thresh):
    plt.plot(pd)
    plt.show(block=False)
    plt.pause(0.5)
    plt.close()
    known = data.known
    # print(data.step1_labels)
    file_split_num = data.file_split_num

    prediction = []
    file_pd = []
    idx = 0

    for i in file_split_num:
        file_pd = np.array(pd[0][idx:idx+i*data.n_mels])
        plt.plot(file_pd)
        plt.show(block=False)
        plt.pause(2)
        plt.close()
        # print(file_pd)
        # print(file_pd>thresh)
        # print(file_pd[file_pd>thresh])
        # print(file_pd.size)
        acc_0_part = file_pd[file_pd>thresh].size/file_pd.size
        for thresh2 in np.arange(0,0.8,0.05):
            if acc_0_part > thresh2:
                prediction.append(0)
            else:
                prediction.append(1)
            # print(acc_0_part)

        idx += i * data.n_mels

    prediction = np.array(prediction)
    acc = prediction[prediction == data.step1_labels].size/prediction.size

    print('-------------------known : ', known, 'acc : ', acc )


