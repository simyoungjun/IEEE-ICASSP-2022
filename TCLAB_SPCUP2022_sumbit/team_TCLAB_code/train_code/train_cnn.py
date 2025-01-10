# %
from tensorflow.python.client import device_lib
device_lib.list_local_devices()

# %
from class_new import *
import class_new
import importlib

importlib.reload(class_new)


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, recall_score, accuracy_score, precision_score

from tensorflow.keras.layers import Conv2D, MaxPool2D, Dense, Flatten, Dropout, UpSampling2D, Input, Convolution2D, \
    Reshape

known_volume_path = 'C:/Users/sim/연구실/음성합성분류 전자공학회/part2_full_train_4X5000'
unknown_volume_path = 'C:/Users/sim/연구실/음성합성분류 전자공학회//unseen_noisy_4X1000'

rs = 10
known_path, known_labels = part2_file_path_list(known_volume_path, label=True)
unknown_path, unknown_labels = part2_file_path_list(unknown_volume_path, label=False)
# %
# print(known_path)
print(known_labels.shape)
print(unknown_labels.shape)

all_path = np.concatenate((known_path, unknown_path), axis=0)
all_labels = np.concatenate((known_labels, unknown_labels), axis=0)
print(all_labels.shape)

##train set
# print('raw train_set_num :',len(labels))
X_train_path, X_test_path, y_train_raw, y_test_raw = train_test_split(all_path,
                                                                      all_labels, test_size=0.2,
                                                                      stratify=all_labels, random_state=rs)
# %
n_mels = 64
samplint_rate = 16000
# train = classes.data(X_train_path, y_train_raw, n_mels=n_mels, known = True)
# # X_train,y_train = train.extract_mel(sampling_rate,n_mels)

# test = classes.data(X_test_path, y_test_raw, n_mels=n_mels, known = True)
# # X_test,y_test = test.extract_mel(sampling_rate,n_mels)

# unseen = data(unknown_path, unknown_labels, n_mels=n_mels, known = False)

# n = -1

train = data(X_train_path, y_train_raw, n_mels=n_mels, known=True)

test = data(X_test_path, y_test_raw, n_mels=n_mels, known=True)

train.extract_mel()
test.extract_mel()

train.reshape_data()
test.reshape_data()


X_train = train.X_reshaped
X_test = test.X_reshaped
y_train = train.y_reshaped
y_test = test.y_reshaped
# %
np.save('x_train_19200_to_mel',X_train)
np.save('x_test_4800_to_mel',X_test)
np.save('y_train_19200_to_mel',y_train)
np.save('y_test_4800_to_mel',y_test)


# %
for n_mels in [128]:
    for hop_length in [512]:
        for n_fft in [2048]:
            from tensorflow.keras.models import Sequential
            from tensorflow.keras.layers import Flatten, Conv2D, MaxPool2D
            from tensorflow.keras.callbacks import EarlyStopping
            from tensorflow.keras.callbacks import ModelCheckpoint

            print((X_train.shape[1:]))
            model = Sequential()

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
                                         + "part2_mel_feature64_label_6_model.h5",
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
            # model.save("./best_model_noise" + "/" + str(rs) + "_" + "part2_label6_spcup_save_last.h5")







