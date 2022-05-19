
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
# from PIL import Image
import pathlib
import csv
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from tensorflow.keras import layers
import pandas as pd
from tensorflow.keras.utils import to_categorical
import statsmodels.api as sm
import time
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from numba import core

class data:
    file_split_num = []

    def __init__(self, FilePathList, labels, sampling_rate=16000, n_mels=64, known=True):
        self.FilePathList = FilePathList
        self.labels = labels
        self.known = known
        self.sampling_rate = sampling_rate
        self.n_mels = n_mels

    def extract_mel(self):
        X_mel_aug = []
        y_labels_aug = []
        sampling_rate = self.sampling_rate
        self.file_split_num = []
        for i, filepath in enumerate(self.FilePathList):
            # fig, ax = plt.subplots()
            y, sr = librosa.load(filepath, mono=True, sr=sampling_rate)
            index_f = 0
            for j in range(y.size // sampling_rate):
                X_aug_seg = y[index_f:index_f + sampling_rate]

                S = librosa.feature.melspectrogram(y=X_aug_seg, sr=sampling_rate, n_mels=self.n_mels, fmax=8000)
                # S_dB = S
                S_dB = librosa.power_to_db(S, ref=np.max)
                X_mel_aug.append(S_dB)

                if self.known ==True:
                    y_labels_aug.append(self.labels[i])

                index_f = index_f + sampling_rate
            self.file_split_num.append(j + 1)
            #         X_aug_seg = np.expand_dims(X_aug_seg, axis=0)
            #         X_aug.append(X_aug_seg)
        # X_aug = np.concatenate(X_aug,axis = 0)
        X_split = np.array(X_mel_aug)
        #         print('train set.shape : ',X_split.shape)
        #         print('y_train.shape',y_split.shape )
        self.X_split = X_split

        if self.known ==True:
            y_split = np.array(y_labels_aug)
            self.y_split = y_split
        else:
            y_split = []


    def extract_mfcc(self):
        sampling_rate = self.sampling_rate
        X_mel_aug = []
        y_labels_aug = []
        self.file_split_num = []
        for i, filepath in enumerate(self.FilePathList):
            # fig, ax = plt.subplots()
            y, sr = librosa.load(filepath, mono=True, sr=self.sampling_rate)
            index_f = 0
            for j in range(y.size // sampling_rate):
                X_aug_seg = y[index_f:index_f + sampling_rate]

                S = librosa.feature.melspectrogram(y=X_aug_seg, sr=sampling_rate, n_mels=self.n_mels, fmax=8000)
                S_dB = librosa.power_to_db(S, ref=np.max)
                mfcc = librosa.feature.mfcc(S=S_dB, n_mfcc=20)
                X_mel_aug.append(mfcc)

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
            self.labels_flatten = np.zeros(self.y_split.size)
        else:
            self.labels_flatten = np.ones(self.y_split.size)


    def extract_cqt(self):
        fmin = librosa.midi_to_hz(36)
        hop_length = 512
        X_cqt_aug = []
        y_labels_aug = []
        sampling_rate = self.sampling_rate
        self.file_split_num = []
        for i, filepath in enumerate(self.FilePathList):
            # fig, ax = plt.subplots()
            y, sr = librosa.load(filepath, mono=True, sr=sampling_rate)
            index_f = 0
            for j in range(y.size // sampling_rate):
                X_aug_seg = y[index_f:index_f + sampling_rate]


                C = librosa.cqt(X_aug_seg, sr=sr, fmin=fmin, n_bins=64, hop_length=hop_length)
                logC = librosa.amplitude_to_db(np.abs(C))
                X_cqt_aug.append(logC)


                if self.known == True:
                    y_labels_aug.append(self.labels[i])

                index_f = index_f + sampling_rate
            self.file_split_num.append(j + 1)
            #         X_aug_seg = np.expand_dims(X_aug_seg, axis=0)
            #         X_aug.append(X_aug_seg)
        # X_aug = np.concatenate(X_aug,axis = 0)
        X_split = np.array(X_cqt_aug)
        #         print('train set.shape : ',X_split.shape)
        #         print('y_train.shape',y_split.shape )
        self.X_split_cqt = X_split

        if self.known == True:
            y_split = np.array(y_labels_aug)
            self.y_split_cqt = y_split
        else:
            y_split = []

    def reshape_data(self):
        self.X_reshaped = self.X_split.reshape(self.X_split.shape[0], self.X_split.shape[1], self.X_split.shape[2], 1)
        print('X_train.shape : ', self.X_reshaped.shape)

        if self.known == True:
            self.y_reshaped = to_categorical(self.y_split)
        # print(y_test.shape)
            print('y_train.shape', self.y_reshaped.shape)

    def reshape_data_cqt_mel(self):

        self.X_reshaped1 = self.X_split.reshape(self.X_split.shape[0], self.X_split.shape[1], self.X_split.shape[2], 1)
        self.X_reshaped2 = self.X_split_cqt.reshape(self.X_split.shape[0], self.X_split.shape[1], self.X_split.shape[2], 1)

        scaler = StandardScaler()
        self.X_scaled1 = scaler.fit_transform(self.X_reshaped1.reshape(-1, self.X_reshaped1.shape[-1])).reshape(self.X_reshaped1.shape)
        self.X_scaled2 = scaler.fit_transform(self.X_reshaped2.reshape(-1, self.X_reshaped2.shape[-1])).reshape(self.X_reshaped2.shape)
        self.X_mel_cqt = np.concatenate((self.X_scaled1, self.X_scaled2), axis=3)

        # print('data.X_mel_cqt.shape : ', self.X_mel_cqt.shape)

        if self.known == True:
            self.y_reshaped = to_categorical(self.y_split)
        # print(y_test.shape)
            print('y_train.shape', self.y_reshaped.shape)

    def flatten(self):
        self.X_flattened = self.X_split.reshape(self.X_split.shape[0], -1)
        print('X_split to X_flattend.shape : ', self.X_flattened.shape)
        if self.known == True:
            self.labels_flatten = np.zeros(self.y_split.size)
        else:
            self.labels_flatten = np.ones(self.y_split.size)
        return self.X_flattened

    def std_scale(self):
        scaler = StandardScaler()
        self.X_split = scaler.fit_transform(self.X_split.reshape(-1, self.X_split.shape[-1])).reshape(self.X_split.shape)

    def min_max_scale(self):
        scaler = MinMaxScaler()
        self.X_min_max_scaled = scaler.fit_transform(self.X_flattened)
        return self.X_min_max_scaled

    #         return self.X_reshaped, self.y_reshaped

    def new_labels(self):
        if self.known == True:
            self.step1_split_labels = np.zeros(self.y_split.size)
            self.step1_labels = np.zeros(self.labels.size)
        else:
            self.step1_split_labels = np.ones(self.y_split.size)
            self.step1_labels = np.ones(self.labels.size)


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
    pd_label = 0
    for dirName, subdirList, fileList in os.walk(volume_path):
        for filename in fileList:
            if '.csv' not in filename:
                #             print(filename)
                FilePathList.append(volume_path + '/' + filename)
            else:
                pd_label = pd.read_csv(volume_path + '/' + filename)

    labels = np.array(pd_label['algorithm'])
    return FilePathList, labels

def eval_file_path_list(volume_path):
    FilePathList = []

    for dirName, subdirList, fileList in os.walk(volume_path):
        for filename in fileList:
            if '.csv' not in filename:
                #             print(filename)
                FilePathList.append(volume_path + '/' + filename)
            else:
                pd_label = volume_path + '/' + filename


    # labels = np.array(pd_label['algorithm'])
    return FilePathList, pd_label


def part2_file_path_list(volume_path, label = False):
    FilePathList = []
    labels = []

    if label ==True:
        for dirName, subdirList, fileList in os.walk(volume_path):
            for filename in fileList:
                if '.csv' not in filename:
                    #             print(filename)
                    FilePathList.append(volume_path + '/' + filename)
                    labels.append(int(filename[0]))
                # else:
                #     pd_label = pd.read_csv(volume_path + '/' + filename)
    else:
        for dirName, subdirList, fileList in os.walk(volume_path):
            for filename in fileList:
                if '.csv' not in filename:
                    #             print(filename)
                    FilePathList.append(volume_path + '/' + filename)
                    labels.append(5)
                # else:
                #     pd_label = pd.read_csv(volume_path + '/' + filename)

    return np.array(FilePathList), np.array(labels)

def part1_file_path_list(volume_path, label = False):
    FilePathList = []
    labels = []

    if label ==True:
        for dirName, subdirList, fileList in os.walk(volume_path):
            for filename in fileList:
                if '.csv' not in filename:
                    #             print(filename)
                    FilePathList.append(volume_path + '/' + filename)
                else:
                    pd_label = pd.read_csv(volume_path + '/' + filename)

        labels = np.array(pd_label['algorithm'])
    else:
        for dirName, subdirList, fileList in os.walk(volume_path):
            for filename in fileList:
                if '.csv' not in filename:
                    #             print(filename)
                    FilePathList.append(volume_path + '/' + filename)
                    labels.append(5)


    return np.array(FilePathList), np.array(labels)

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
        # scaler = StandardScaler()
        # self.reshaped_data = scaler.fit_transform(self.reshaped_data)
        mm_scaler = MinMaxScaler()
        self.reshaped_data = mm_scaler.fit_transform(self.reshaped_data)

    def make_pdf(self):
        var_type = 'u' * self.n_mels

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
    print(density.dens)
    print(density.bw)

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