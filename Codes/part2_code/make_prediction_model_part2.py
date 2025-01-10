#!/usr/bin/env python
# coding: utf-8

# In[4]:


import librosa
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers

#volume_path = 'D:\spcup\spcup_2022_training_part1'
volume_path = './part2_full_train_4X5000'
# volume_path = 'C:/Users/GJ/Desktop/연구실/2022SPCUP/spcup_2022_training_part1'

FilePathList = []
labels = []


for dirName, subdirList, fileList in os.walk(volume_path):
    for filename in fileList:
        if '.csv' not in filename:
            # print(filename)
            FilePathList.append(volume_path + '/' + filename)
            labels.append(int(filename[0]))
        else:
            pd_label = pd.read_csv(volume_path + '/' + filename)

labels = np.array(labels)
#labels = np.array(pd_label['algorithm'])

##파일마다 1초씩 슬라이싱 해서 data augmentation + 마지막에 파일 별로 score 계산하기 위한 preprocess
# X_aug = []

sampling_rate = 16000
n_mels = 64
test_accuracy_all = []


# In[19]:


#for rs in list(range(43, 52)):
rs = 42

X_train_path, X_test_path, y_train_raw, y_test_raw = train_test_split(np.array(FilePathList), labels, test_size=0.2, stratify=labels, random_state=rs)

##train set
print('raw train_set_num :', len(y_train_raw))

y_labels_aug = []
X_mel_aug = []
for i, filepath in enumerate(X_train_path):
    # fig, ax = plt.subplots()
    y, sr = librosa.load(filepath, mono=True, sr=sampling_rate)
    index_f = 0
    for j in range(y.size // sampling_rate):
        X_aug_seg = y[index_f:index_f + sampling_rate]

        S = librosa.feature.melspectrogram(y=X_aug_seg, sr=sampling_rate, n_mels=n_mels, fmax=8000)
        S_dB = librosa.power_to_db(S, ref=np.max)
        X_mel_aug.append(S_dB)

        y_labels_aug.append(y_train_raw[i])

        index_f = index_f + sampling_rate
        #         X_aug_seg = np.expand_dims(X_aug_seg, axis=0)
        #         X_aug.append(X_aug_seg)
# X_aug = np.concatenate(X_aug,axis = 0)
X_train1 = np.array(X_mel_aug)
y_train1 = np.array(y_labels_aug)
print('train set.shape : ', X_train1.shape)
print('y_train.shape', y_train1.shape)


# In[20]:


##test set
print('raw test_set_num :', len(y_test_raw))
y_labels_aug = []
X_mel_aug = []
file_split_num = []
for i, filepath in enumerate(X_test_path):
    # fig, ax = plt.subplots()
    y, sr = librosa.load(filepath, mono=True, sr=sampling_rate)
    index_f = 0
    for j in range(y.size // sampling_rate):
        X_aug_seg = y[index_f:index_f + sampling_rate]

        S = librosa.feature.melspectrogram(y=X_aug_seg, sr=sampling_rate, n_mels=n_mels, fmax=8000)
        #S = librosa.feature.melspectrogram(y=X_aug_seg, sr=sampling_rate, n_fft=512, hop_length=512, n_mels=n_mels, fmax=8000)
        S_dB = librosa.power_to_db(S, ref=np.max)
        X_mel_aug.append(S_dB)

        y_labels_aug.append(y_test_raw[i])

        index_f = index_f + sampling_rate
    file_split_num.append(j + 1)

X_test1 = np.array(X_mel_aug)
y_test1 = np.array(y_labels_aug)
print('splited_test set.shape : ', X_test1.shape)
print('y_test.shape : ', y_test1.shape)
print(len(file_split_num))

###Train ,Test data split
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

# X_train, X_test, y_train, y_test = train_test_split(pad_x_arr, label, test_size=0.2, stratify = label, random_state=42)
# X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, stratify = Y, random_state=42)

print('np.unique(y_train)', np.unique(y_train1))

for i in range(5):
    print('전체 데이터 label' + '==', i, '분포 : ', labels.tolist().count(i))
for i in range(5):
    print('train set label' + '==', i, '분포 : ', y_train1.tolist().count(i) / labels.tolist().count(i))
for i in range(5):
    print('test set label' + '==', i, '분포 : ', y_test1.tolist().count(i))

X_train = X_train1.reshape(X_train1.shape[0], X_train1.shape[1], X_train1.shape[2], 1)
X_test = X_test1.reshape(X_test1.shape[0], X_test1.shape[1], X_test1.shape[2], 1)
print('X_train.shape : ', X_train.shape)
print('X_test.shape : ', X_test.shape)

y_train = to_categorical(y_train1)
y_test = to_categorical(y_test1)
# print(y_test.shape)
print('y_train.shape', y_train.shape)
print('y_test.shape : ', y_test.shape)

# print('y : ',y_test[:10])


# In[22]:


## model compile & fit

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Conv2D, MaxPool2D
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import ModelCheckpoint

print((X_train.shape[1:]))
model = Sequential()
'''
#model 1
model.add(layers.BatchNormalization(input_shape=(X_train.shape[1:])))
model.add(layers.Conv2D(filters=64, kernel_size=(5, 5), padding="same", activation='relu'))
model.add(layers.BatchNormalization())
model.add(layers.Conv2D(filters=128, kernel_size=(5, 5), padding="same", activation='relu'))
model.add(layers.BatchNormalization())
model.add(layers.Conv2D(filters=64, kernel_size=(5, 5), padding="same", activation='relu'))
model.add(MaxPool2D(pool_size=(5, 5)))
model.add(Flatten())
model.add(layers.Dense(5, activation='softmax'))
'''
#model 2
model.add(layers.BatchNormalization(input_shape=(X_train.shape[1:])))
model.add(layers.Conv2D(filters=16, kernel_size=(5, 5), padding="same", activation='relu'))
model.add(layers.BatchNormalization())
model.add(layers.Conv2D(filters=32, kernel_size=(5, 5), padding="same", activation='relu'))
model.add(layers.BatchNormalization())
model.add(layers.Conv2D(filters=64, kernel_size=(5, 5), padding="same", activation='relu'))
model.add(MaxPool2D(pool_size=(5, 5)))
model.add(Flatten())
model.add(layers.Dense(5, activation='softmax'))

# 모델 컴파일
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=100, mode='auto')
model_checkpoint = ModelCheckpoint(filepath="E:/spcup2022/best_model" + "/" + str(rs) + "_" + "spcup_best.h5", monitor='val_loss', verbose=1, save_best_only=True)

classifier = model.fit(X_train,
                       y_train,
                       epochs=1000,
                       batch_size=128, callbacks=[early_stopping,model_checkpoint],
                       validation_split=0.2)

model_json = model.to_json()
with open("D:\spcup_files\spcup_my_python_models" + "/" + str(rs) + "_" + "spcup.json", 'w') as json_file:
    json_file.write(model_json)
model.save_weights("D:\spcup_files\spcup_my_python_models" + "/" + str(rs) + "_" + "spcup.h5")
print("Saved model to disk")

plt.plot(classifier.history['accuracy'])
plt.plot(classifier.history['val_accuracy'])
# plt.plot(classifier.history['loss'])
# plt.plot(classifier.history['val_loss'])
plt.legend(['val_accuracy', 'accuracy'], loc='upper left')

test_loss, test_acc = model.evaluate(X_test, y_test, verbose=2)

print('테스트 정확도:', test_acc)

prediction = model.predict(X_test)


# In[ ]:


from scipy.stats import mode

predicted_classes = np.argmax(prediction, axis=1)
f = 0
test_predict = []
for i in file_split_num:
    #     print(predicted_classes[f:f+i])
    test_predict.append(mode(predicted_classes[f:f + i])[0][0])
    f += i

acc_bool = test_predict == y_test_raw
print(test_predict[:100])
print(y_test_raw[:100])
print(acc_bool[:100])
test_accuracy = acc_bool.tolist().count(True) / len(acc_bool)
print(test_accuracy)
print('test_set.size : ', acc_bool.size)

#plt.show()

test_accuracy_all.append(test_accuracy)
print(test_accuracy_all)

import csv

f = open('Accuracy.csv', 'w', newline='')
wr = csv.writer(f)
wr.writerow([1, test_accuracy_all])

f.close()


# In[ ]:




