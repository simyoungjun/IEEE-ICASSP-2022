import keras
from keras import layers
from keras.models import Sequential
from keras.layers import Flatten, Conv2D, MaxPooling2D
from keras.callbacks import EarlyStopping

model = Sequential()
model.add(layers.Conv2D(64, (3, 3), activation='relu', input_shape=(32,32,3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Dense(32, activation='relu'))
model.add(Flatten())
model.add(layers.Dense(32, activation='relu'))
model.add(layers.Dense(5, activation='relu'))
model.add(layers.Activation('softmax'))


model.summary()

