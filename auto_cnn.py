# import import_ipynb
from classes import *
import classes
import tensorflow as tf
import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, recall_score, accuracy_score, precision_score

from tensorflow.keras.layers import Conv2D, MaxPool2D, Dense, Flatten, Dropout, UpSampling2D, Input, Convolution2D, Reshape

known_volume_path = 'C:/Users/GJ/Desktop/연구실/2022SPCUP/spcup_2022_training_part1'
unknown_volume_path = './spcup_2022_unseen'




rs = 10
known_path, known_labels = file_path_list(known_volume_path)
unknown_path, unknown_labels = file_path_list(unknown_volume_path)
##train set
# print('raw train_set_num :',len(labels))
X_train_path, X_test_path, y_train_raw, y_test_raw = train_test_split(np.array(known_path),
                                                                      known_labels, test_size=0.2,
                                                                      stratify = known_labels, random_state=rs)


n_mels = 64

# train = classes.data(X_train_path, y_train_raw, n_mels=n_mels, known = True)
# # X_train,y_train = train.extract_mel(sampling_rate,n_mels)
#
# test = classes.data(X_test_path, y_test_raw, n_mels=n_mels, known = True)
# # X_test,y_test = test.extract_mel(sampling_rate,n_mels)
#
# unseen = data(unknown_path, unknown_labels, n_mels=n_mels, known = False)

n = -1
train = classes.data(X_train_path[:n],y_train_raw[:n],n_mels=n_mels, known = True)

test = classes.data(X_test_path[:n],y_test_raw[:n], n_mels=n_mels, known = True)

unseen =data(unknown_path[:n],unknown_labels[:n], n_mels=n_mels, known = False)


train.reshape_data()
test.reshape_data()
unseen.reshape_data()



'''
input_layer = tf.keras.layers.Input(shape=(input_dim, ))
encoder = tf.keras.layers.Dense(encoding_dim, activation="relu", activity_regularizer=tf.keras.regularizers.l1(learning_rate))(input_layer)
encoder = tf.keras.layers.Dense(encoding_dim, activation="relu")(input_layer)
#encoder = tf.keras.layers.Dropout(0.2)(encoder)
encoder = tf.keras.layers.Dense(hidden_dim_1, activation='relu')(encoder)
encoder = tf.keras.layers.Dense(hidden_dim_2, activation='relu')(encoder)

decoder = tf.keras.layers.Dense(hidden_dim_1, activation='relu')(encoder)
#decoder = tf.keras.layers.Dropout(0.2)(decoder)
decoder = tf.keras.layers.Dense(encoding_dim, activation='relu')(decoder)
decoder = tf.keras.layers.Dense(input_dim, activation='softmax')(decoder)
'''

from tensorflow.keras.layers import Flatten, Conv2D, MaxPool2D
# 인코더
latent_dim = 32
def encoder():

  model = tf.keras.Sequential()
  model.add(Conv2D(5, (3, 3), activation='relu', padding='same', input_shape=(64, 32, 1)))
  # model.add(MaxPool2D((2, 2), padding='same'))
  model.add(Conv2D(10, (3, 3), activation='relu', padding='same'))
  # model.add(MaxPool2D((2, 2), padding='same'))
  model.add(Conv2D(15, (3, 3), activation='relu', padding='same'))
  # model.add(MaxPool2D((2, 2), padding='same'))
  # model.add(Conv2D(20, (3, 3), activation='relu', padding='same'))
  # model.add(MaxPool2D((2, 2), padding='same'))
  model.add(Conv2D(25, (3, 3), activation='relu', padding='same'))
  model.add(Flatten())
  # latent dimension으로 축소됨
  model.add(Dense(latent_dim))

  return model

e_model = encoder()
e_model.summary()

# 디코더
def decoder():

  model = tf.keras.Sequential()
  model.add(Dense(8* 4 * 56, input_shape=(latent_dim,)))
  model.add(Reshape((8, 4, 56)))
  model.add(Conv2D(56, (2, 2), activation='relu', padding='same'))
  model.add(UpSampling2D((2, 2)))
  model.add(Conv2D(56, (2, 2), activation='relu', padding='same'))
  model.add(UpSampling2D((2, 2)))
  model.add(Conv2D(64, (2, 2), activation='relu', padding='same'))
  model.add(UpSampling2D((2, 2)))
  model.add(Dropout(0.5))
  model.add(Conv2D(5, (2, 2), padding='same'))

  return model

d_model = decoder()
d_model.summary()

# 모델을 합쳐줌
input_img = Input(shape=(n_mels, 32, 1))
model = tf.keras.Model(input_img, d_model(e_model(input_img)), name='autoencoder')

# optimizer 설정과 model complie하면서 optimizer와 loss 함수, loss 평가지표로 mae 설정
adam= tf.keras.optimizers.Adam(lr=0.00008, beta_1=0.9)

cp = tf.keras.callbacks.ModelCheckpoint(filepath=str(n_mels)
                                        +'_'+"cnnencoder_best.h5", mode='auto', monitor='val_loss',
                                        verbose=2, save_best_only=True)

early_stop = tf.keras.callbacks.EarlyStopping( monitor='val_loss', min_delta=0.1,
                                              patience=100, verbose=1, mode='min', restore_best_weights=True)

model.compile(optimizer=adam, loss='mean_squared_error', metrics=['mae'])


history = model.fit(train.X_reshaped, train.X_reshaped,
                    batch_size=64,
                    epochs=1000,
                    verbose=2,
                    validation_split= 0.2,
                    shuffle=True)

# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])

plt.title('loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()



test_x_predictions = model.predict(test.X_reshaped)
mse_test = np.mean(np.power(test.X_reshaped - test_x_predictions, 2), axis=1)
# error_df = pd.DataFrame({'Reconstruction_error': mse_test, 'True_class': test.labels})

unseen_x_predictions = model.predict(unseen.X_reshaped)
mse_unseen = np.mean(np.power(unseen.X_reshaped - unseen_x_predictions, 2), axis=1)
# error_df = pd.DataFrame({'Reconstruction_error': mse_unseen, 'True_class': test.labels})

all_mse = np.concatenate((mse_test,mse_unseen),axis = None)

plt.subplot()
plt.plot(all_mse)
plt.show()