# import import_ipynb

import sys
sys.path.append('C:/Users/GJ/PycharmProjects/2022SPCUP')


from class_new import *
import class_new
import importlib
importlib.reload(class_new)

import tensorflow as tf
import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, recall_score, accuracy_score, precision_score

from tensorflow.keras.layers import Conv2D, MaxPool2D, Dense, Flatten, Dropout, UpSampling2D, Input, Convolution2D, Reshape

known_volume_path = './part2_full_train_4X5000'
unknown_volume_path = './unseen_noisy_4X1000'




rs = 10
known_path, known_labels = part2_file_path_list(known_volume_path,True)
unknown_path, unknown_labels = part2_file_path_list(unknown_volume_path,False)

# print(known_path)
print(known_labels.shape)
print(unknown_labels.shape)

all_path = np.concatenate((known_path,unknown_path),axis = 0)
all_labels = np.concatenate((known_labels,unknown_labels),axis = 0)
print(all_labels.shape)


##train set
# print('raw train_set_num :',len(labels))
X_train_path, X_test_path, y_train_raw, y_test_raw = train_test_split(all_path,
                                                                      all_labels, test_size=0.2,
                                                                      stratify = all_labels, random_state=rs)


n_mels = 128
samplint_rate = 16000
# train = classes.data(X_train_path, y_train_raw, n_mels=n_mels, known = True)
# # X_train,y_train = train.extract_mel(sampling_rate,n_mels)
#
# test = classes.data(X_test_path, y_test_raw, n_mels=n_mels, known = True)
# # X_test,y_test = test.extract_mel(sampling_rate,n_mels)
#
# unseen = data(unknown_path, unknown_labels, n_mels=n_mels, known = False)

# n = -1
train = data(X_train_path,y_train_raw,n_mels=n_mels, known = True)

test = data(X_test_path,y_test_raw, n_mels=n_mels, known = True)

train.extract_mel()
test.extract_mel()



train.reshape_data()
test.reshape_data()



X_train = train.X_reshaped
X_test = test.X_reshaped
y_train = train.y_reshaped
y_test = test.y_reshaped

for n_mels in [64]:
    for hop_length in [1024]:
        for n_fft in [2048]:
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
            model.add(layers.Dense(6, activation=None))
            model.add(layers.Activation('softmax'))


            # 모델 컴파일
            model.compile(optimizer='adam',
                          loss='categorical_crossentropy',
                          metrics=['accuracy'])


            callbacks = [EarlyStopping(monitor='val_loss', min_delta=0, patience=100, mode='auto',restore_best_weights=True),
                         ModelCheckpoint(filepath=str(n_mels)+"_"+str(hop_length)+"_"+ str(rs) + "_"
                                         + "part2_mel_feature128_label_6_model.h5",
                                         monitor='val_loss', verbose=0, save_best_only=True)]

            classifier = model.fit(X_train,
                                   y_train,
                                   epochs=1000,
                                   batch_size=128, callbacks=callbacks,
                                   validation_split=0.2)

#             model_json = model.to_json()
#             with open("./models_log" + "/" + str(rs) + "_" + "spcup.json", 'w') as json_file:
#                 json_file.write(model_json)
            #             model.save_weights("./models_log" + "/" + str(rs) + "_" + "spcup.h5")
            print("Saved model to disk")


            #             #### json 모델 load
            #             from keras.models import model_from_json
            #             json_file = open("model.json", "r")
            #             loaded_model_json = json_file.read()
            #             json_file.close()
            #             loaded_model = model_from_json(loaded_model_json)

            plt.plot(classifier.history['accuracy'])
            plt.plot(classifier.history['val_accuracy'])
            # plt.plot(classifier.history['loss'])
            # plt.plot(classifier.history['val_loss'])
            plt.legend([ 'accuracy','val_accuracy'], loc = 'upper left')
            plt.show()

            test_loss, test_acc = model.evaluate(X_test,  y_test, verbose=2)

            print('테스트 정확도:', test_acc)

            prediction = model.predict(X_test)




            #######################segment들 합쳐서 파일당 accuracy 계산
            from scipy.stats import mode
            predicted_classes = np.argmax(prediction, axis = 1)
            f = 0
            test_predict=[]
            for i in file_split_num:
            #     print(predicted_classes[f:f+i])
                test_predict.append(mode(predicted_classes[f:f+i])[0][0])
                f+=i

            acc_bool = test_predict==y_test_raw
            #             print(test_predict[:100])
            #             print(y_test_raw[:100])
            #             print(acc_bool[:100])
            test_accuracy = acc_bool.tolist().count(True)/len(acc_bool)
            print('************ mel_spectogram')
            print('************ file test_accuracy : ', test_accuracy, '  n_mels : ', n_mels,
                  ', hop_length = ', hop_length,' , n_fft : ',n_fft)
            print('************ X_train.shape : ', X_train.shape)
            print('test_set.size : ', acc_bool.size)
            print('\n\n\n')




