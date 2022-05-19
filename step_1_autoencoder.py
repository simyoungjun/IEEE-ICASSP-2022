# import import_ipynb
from classes import *
import classes
import tensorflow as tf
import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, recall_score, accuracy_score, precision_score



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

n = 10
n_mels = 64
# train = classes.data(X_train_path,y_train_raw,n_mels=n_mels, known = True)
#
# test = classes.data(X_test_path,y_test_raw, n_mels=n_mels, known = True)
#
# unknown =data(unknown_path,unknown_labels, n_mels=n_mels, known = False)

train = classes.data(X_train_path[:n],y_train_raw[:n],n_mels=n_mels, known = True)

test = classes.data(X_test_path[:n],y_test_raw[:n], n_mels=n_mels, known = True)

unknown =data(unknown_path[:n],unknown_labels[:n], n_mels=n_mels, known = False)

train_au = train.flatten()
test_au = test.flatten()
unseen_au = unknown.flatten()

train.min_max_scale()
test.min_max_scale()
unknown.min_max_scale()

nb_epoch = 10000
batch_size = 64
input_dim = train_au.shape[1]
encoding_dim = 32
hidden_dim_1 = int(encoding_dim/2)
hidden_dim_2 = 8
hidden_dim_3 =4

input_layer = tf.keras.layers.Input(shape=(input_dim, ))
encoder = tf.keras.layers.Dense(encoding_dim, activation="relu", activity_regularizer=tf.keras.regularizers.l2(0.1))(input_layer)
encoder= tf.keras.layers.Dropout(0.2)(encoder)
encoder = tf.keras.layers.Dense(hidden_dim_1, activation='relu')(encoder)
encoder = tf.keras.layers.Dense(hidden_dim_2, activation='relu')(encoder)
encoder = tf.keras.layers.Dense(hidden_dim_3, activation='relu')(encoder)

decoder = tf.keras.layers.Dense(hidden_dim_3, activation='relu')(encoder)
decoder = tf.keras.layers.Dense(hidden_dim_2, activation='relu')(decoder)
decoder = tf.keras.layers.Dense(hidden_dim_1, activation='relu')(decoder)
decoder=tf.keras.layers.Dropout(0.2)(decoder)
decoder = tf.keras.layers.Dense(encoding_dim, activation='relu')(decoder)
decoder = tf.keras.layers.Dense(input_dim, activation='relu')(decoder)

autoencoder = tf.keras.Model(inputs=input_layer, outputs=decoder)
autoencoder.summary()

cp = tf.keras.callbacks.ModelCheckpoint(filepath=str(n_mels)+'_'+str(encoding_dim)+'_'+str(hidden_dim_2)
                                        +'_'+"autoencoder_best.h5", mode='auto', monitor='val_loss',
                                        verbose=2, save_best_only=True)

early_stop = tf.keras.callbacks.EarlyStopping( monitor='val_loss', min_delta=0.1,
                                              patience=1000, verbose=1, mode='min', restore_best_weights=True)


optimizer = keras.optimizers.Adam(lr=0.1)
autoencoder.compile(metrics=['accuracy'], loss='mean_squared_error', optimizer=optimizer )

history = autoencoder.fit(train_au, train_au, epochs=nb_epoch,
                         batch_size=batch_size, shuffle=True,
                          validation_split= 0.2, verbose=1,
                          callbacks=[cp, early_stop] ).history


plt.plot(history['loss'], linewidth=2, label='Train')
plt.plot(history['val_loss'], linewidth=2, label='val')
plt.legend(loc='upper right')
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch') #plt.ylim(ymin=0.70,ymax=1) plt.show()
plt.show()

test_x_predictions = autoencoder.predict(test_au)
mse_test = np.sqrt(np.mean(np.power(test_au - test_x_predictions, 2), axis=1))
# error_df = pd.DataFrame({'Reconstruction_error': mse_test, 'True_class': test.labels})


unseen_x_predictions = autoencoder.predict(unseen_au)
mse_unseen = np.sqrt(np.mean(np.power(unseen_au - unseen_x_predictions, 2), axis=1))
# error_df = pd.DataFrame({'Reconstruction_error': mse_unseen, 'True_class': test.labels})

all_mse = np.concatenate((mse_test,mse_unseen),axis = None)

plt.subplot()
plt.plot(all_mse)
plt.show()
