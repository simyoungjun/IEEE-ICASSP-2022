import tensorflow
from tensorflow import keras
from tensorflow.keras import layers
from keras.models import Sequential
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPool2D
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout

# def CNNmodel():
#
#     model = Sequential()
#     model.add(tensorflow.keras.layers.BatchNormalization(input_shape=(32, 64, 1)))
#     model.add(Conv2D(filters=16, kernel_size=(3, 3), padding="same", activation='relu')) #,input_shape=(32, 64, 1)
#     model.add(tensorflow.keras.layers.BatchNormalization())
#     model.add(Conv2D(filters=32, kernel_size=(3, 3), padding="same", activation='relu'))
#     model.add(tensorflow.keras.layers.BatchNormalization())
#     model.add(Conv2D(filters=64, kernel_size=(3, 3), padding="same", activation='relu'))
#     model.add(tensorflow.keras.layers.BatchNormalization())
#     model.add(MaxPool2D(pool_size=(3, 3)))
#     model.add(Flatten())
#     model.add(Dense(2, activation='softmax'))
#
#     model.summary()
#     return model

model = Sequential()
model.add(tensorflow.keras.layers.BatchNormalization(input_shape=(32, 64, 1)))
model.add(Conv2D(filters=16, kernel_size=(3, 3), padding="same", activation='relu')) #,input_shape=(32, 64, 1)
model.add(tensorflow.keras.layers.BatchNormalization())
model.add(Conv2D(filters=32, kernel_size=(3, 3), padding="same", activation='relu'))
model.add(tensorflow.keras.layers.BatchNormalization())
model.add(Conv2D(filters=64, kernel_size=(3, 3), padding="same", activation='relu'))
model.add(tensorflow.keras.layers.BatchNormalization())
model.add(MaxPool2D(pool_size=(3, 3)))
model.add(Flatten())
model.add(Dense(2, activation='softmax'))

model.summary()







#
# def CNNmodel2():
#     model = Sequential()
#     model.add(tensorflow.keras.layers.BatchNormalization(input_shape=(32, 64, 1)))
#     model.add(Conv2D(filters=64, kernel_size=(3, 3),padding="same", activation='relu')) #,input_shape=(32, 64, 1)
#     model.add(tensorflow.keras.layers.BatchNormalization())
#     model.add(Conv2D(filters=128, kernel_size=(3, 3), padding="same", activation='relu'))
#     model.add(tensorflow.keras.layers.BatchNormalization())
#     model.add(MaxPool2D(pool_size=(2, 2)))
#     model.add(Flatten())
#     model.add(Dense(2, activation='softmax'))
#     model.summary()
#
#     return model

model = Sequential()
model.add(tensorflow.keras.layers.BatchNormalization(input_shape=(32, 64, 1)))
model.add(Conv2D(filters=64, kernel_size=(3, 3),padding="same", activation='relu')) #,input_shape=(32, 64, 1)
model.add(tensorflow.keras.layers.BatchNormalization())
model.add(Conv2D(filters=128, kernel_size=(3, 3), padding="same", activation='relu'))
model.add(tensorflow.keras.layers.BatchNormalization())
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(2, activation='softmax'))
model.summary()

model = Sequential()
model.add(Conv2D(filters=32, kernel_size=(6, 6),padding="same", activation='relu')) #,input_shape=(32, 64, 1)
model.add(Conv2D(filters=64, kernel_size=(5, 5), activation='relu')) #,input_shape=(32, 64, 1)
model.add(MaxPool2D(pool_size=(2, 2)))


#
# def CNNmodel3():
#     model = Sequential()
#     model.add(tensorflow.keras.layers.BatchNormalization(input_shape=(32, 64, 1)))
#     model.add(Conv2D(filters=128, kernel_size=(3, 3),padding="same", activation='relu'))  # 32 64   /  3,557   / 18,21 / 8, 59 , input_shape=(32, 64, 1)
#     model.add(tensorflow.keras.layers.BatchNormalization())
#     model.add(MaxPool2D(padding="same",pool_size=(3, 3)))
#     model.add(tensorflow.keras.layers.BatchNormalization())
#     model.add(Conv2D(filters=256, kernel_size=(3, 3), padding="same", activation='relu'))
#     model.add(tensorflow.keras.layers.BatchNormalization())
#     model.add(MaxPool2D(padding="same",pool_size=(3, 3)))
#     model.add(tensorflow.keras.layers.BatchNormalization())
#     model.add(Flatten())
#     model.add(Dropout(0.25))
#     #model.add(Dense(20))
#     model.add(Dense(2, activation='softmax'))
#     model.summary()
#
#     return model
#
print('\n\n3')
model = Sequential()
model.add(tensorflow.keras.layers.BatchNormalization(input_shape=(32, 64, 1)))
model.add(Conv2D(filters=128, kernel_size=(3, 3),padding="same", activation='relu'))  # 32 64   /  3,557   / 18,21 / 8, 59 , input_shape=(32, 64, 1)
model.add(tensorflow.keras.layers.BatchNormalization())
model.add(MaxPool2D(padding="same",pool_size=(3, 3)))
model.add(tensorflow.keras.layers.BatchNormalization())
model.add(Conv2D(filters=256, kernel_size=(3, 3), padding="same", activation='relu'))
model.add(tensorflow.keras.layers.BatchNormalization())
model.add(MaxPool2D(padding="same",pool_size=(3, 3)))
model.add(tensorflow.keras.layers.BatchNormalization())
model.add(Flatten())
model.add(Dropout(0.25))
#model.add(Dense(20))
model.add(Dense(2, activation='softmax'))
model.summary()

# def CNNmodel4():
#     model = Sequential()
#     model.add(tensorflow.keras.layers.BatchNormalization(input_shape=(32, 64, 1)))
#     model.add(Conv2D(filters=64, kernel_size=(3, 3),padding="same", activation='relu')) #,input_shape=(32, 64, 1)
#     model.add(tensorflow.keras.layers.BatchNormalization())
#     model.add(Conv2D(filters=128, kernel_size=(3, 3), padding="same", activation='relu'))
#     model.add(tensorflow.keras.layers.BatchNormalization())
#     model.add(Conv2D(filters=256, kernel_size=(3, 3), padding="same", activation='relu'))
#     model.add(tensorflow.keras.layers.BatchNormalization())
#     model.add(MaxPool2D(pool_size=(3, 3)))
#     model.add(Flatten())
#     model.add(Dense(2, activation='softmax'))
#     model.summary()
#
#     return model
#

print('\n\n4')

model = Sequential()
model.add(tensorflow.keras.layers.BatchNormalization(input_shape=(32, 64, 1)))
model.add(Conv2D(filters=64, kernel_size=(3, 3),padding="same", activation='relu')) #,input_shape=(32, 64, 1)
model.add(tensorflow.keras.layers.BatchNormalization())
model.add(Conv2D(filters=128, kernel_size=(3, 3), padding="same", activation='relu'))
model.add(tensorflow.keras.layers.BatchNormalization())
model.add(Conv2D(filters=256, kernel_size=(3, 3), padding="same", activation='relu'))
model.add(tensorflow.keras.layers.BatchNormalization())
model.add(MaxPool2D(pool_size=(3, 3)))
model.add(Flatten())
model.add(Dense(2, activation='softmax'))
model.summary()

# def CNNmodel5():
#     model = Sequential()
#     model.add(tensorflow.keras.layers.BatchNormalization(input_shape=(32, 64, 1)))
#     model.add(Conv2D(filters=64, kernel_size=(3, 3),padding="same", activation='relu'))
#     model.add(Conv2D(filters=128, kernel_size=(3, 3),padding="same", activation='relu'))#,input_shape=(32, 64, 1)
#     model.add(tensorflow.keras.layers.BatchNormalization())
#     model.add(Conv2D(filters=256, kernel_size=(3, 3), padding="same", activation='relu'))
#     model.add(Conv2D(filters=128, kernel_size=(3, 3), padding="same", activation='relu'))
#     model.add(tensorflow.keras.layers.BatchNormalization())
#     model.add(MaxPool2D(pool_size=(2, 2)))
#     model.add(Flatten())
#     model.add(Dense(2, activation='softmax'))
#     model.summary()
#
#     return model
#
print('\n\n5')
model = Sequential()
model.add(tensorflow.keras.layers.BatchNormalization(input_shape=(32, 64, 1)))
model.add(Conv2D(filters=64, kernel_size=(3, 3),padding="same", activation='relu'))
model.add(Conv2D(filters=128, kernel_size=(3, 3),padding="same", activation='relu'))#,input_shape=(32, 64, 1)
model.add(tensorflow.keras.layers.BatchNormalization())
model.add(Conv2D(filters=256, kernel_size=(3, 3), padding="same", activation='relu'))
model.add(Conv2D(filters=128, kernel_size=(3, 3), padding="same", activation='relu'))
model.add(tensorflow.keras.layers.BatchNormalization())
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(2, activation='softmax'))
model.summary()

# def CNNmodel6():
#     model = Sequential()
#     model.add(tensorflow.keras.layers.BatchNormalization(input_shape=(32, 64, 1)))
#     model.add(Conv2D(filters=64, kernel_size=(3, 3), padding="same", activation='relu'))  # ,input_shape=(32, 64, 1)
#     model.add(tensorflow.keras.layers.BatchNormalization())
#     model.add(Conv2D(filters=128, kernel_size=(3, 3), padding="same", activation='relu'))
#     model.add(tensorflow.keras.layers.BatchNormalization())
#     model.add(MaxPool2D(pool_size=(2, 2),padding='same'))
#     model.add(Conv2D(filters=256, kernel_size=(3, 3), padding="same", activation='relu'))  # ,input_shape=(32, 64, 1)
#     model.add(tensorflow.keras.layers.BatchNormalization())
#     model.add(Conv2D(filters=128, kernel_size=(3, 3), padding="same", activation='relu'))
#     model.add(tensorflow.keras.layers.BatchNormalization())
#     model.add(MaxPool2D(pool_size=(2, 2)))
#     model.add(Flatten())
#     model.add(Dense(2, activation='softmax'))
#     model.summary()
#
#     return model

print('\n\n6')
model = Sequential()
model.add(tensorflow.keras.layers.BatchNormalization(input_shape=(32, 64, 1)))
model.add(Conv2D(filters=64, kernel_size=(3, 3), padding="same", activation='relu'))  # ,input_shape=(32, 64, 1)
model.add(tensorflow.keras.layers.BatchNormalization())
model.add(Conv2D(filters=128, kernel_size=(3, 3), padding="same", activation='relu'))
model.add(tensorflow.keras.layers.BatchNormalization())
model.add(MaxPool2D(pool_size=(2, 2),padding='same'))
model.add(Conv2D(filters=256, kernel_size=(3, 3), padding="same", activation='relu'))  # ,input_shape=(32, 64, 1)
model.add(tensorflow.keras.layers.BatchNormalization())
model.add(Conv2D(filters=128, kernel_size=(3, 3), padding="same", activation='relu'))
model.add(tensorflow.keras.layers.BatchNormalization())
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(2, activation='softmax'))
model.summary()

# def CNNmodel7():
#     model = Sequential()
#     model.add(tensorflow.keras.layers.BatchNormalization(input_shape=(32, 64, 1)))
#     model.add(Conv2D(filters=64, kernel_size=(3, 3),padding="same", activation='relu')) #,input_shape=(32, 64, 1)
#     model.add(tensorflow.keras.layers.BatchNormalization())
#     model.add(MaxPool2D(pool_size=(2, 2),padding='same'))
#     model.add(Conv2D(filters=128, kernel_size=(3, 3), padding="same", activation='relu'))  # ,input_shape=(32, 64, 1)
#     model.add(tensorflow.keras.layers.BatchNormalization())
#     model.add(MaxPool2D(pool_size=(2, 2), padding='same'))
#     model.add(Conv2D(filters=256, kernel_size=(3, 3), padding="same", activation='relu'))  # ,input_shape=(32, 64, 1)
#     model.add(tensorflow.keras.layers.BatchNormalization())
#     model.add(MaxPool2D(pool_size=(2, 2), padding='same'))
#     model.add(Conv2D(filters=128, kernel_size=(3, 3), padding="same", activation='relu'))  # ,input_shape=(32, 64, 1)
#     model.add(tensorflow.keras.layers.BatchNormalization())
#     model.add(MaxPool2D(pool_size=(2, 2), padding='same'))
#     model.add(Flatten())
#     model.add(Dense(2, activation='softmax'))
#     model.summary()
#
#     return model

print('\n\n7')
model = Sequential()
model.add(tensorflow.keras.layers.BatchNormalization(input_shape=(32, 64, 1)))
model.add(Conv2D(filters=64, kernel_size=(3, 3),padding="same", activation='relu')) #,input_shape=(32, 64, 1)
model.add(tensorflow.keras.layers.BatchNormalization())
model.add(MaxPool2D(pool_size=(2, 2),padding='same'))
model.add(Conv2D(filters=128, kernel_size=(3, 3), padding="same", activation='relu'))  # ,input_shape=(32, 64, 1)
model.add(tensorflow.keras.layers.BatchNormalization())
model.add(MaxPool2D(pool_size=(2, 2), padding='same'))
model.add(Conv2D(filters=256, kernel_size=(3, 3), padding="same", activation='relu'))  # ,input_shape=(32, 64, 1)
model.add(tensorflow.keras.layers.BatchNormalization())
model.add(MaxPool2D(pool_size=(2, 2), padding='same'))
model.add(Conv2D(filters=128, kernel_size=(3, 3), padding="same", activation='relu'))  # ,input_shape=(32, 64, 1)
model.add(tensorflow.keras.layers.BatchNormalization())
model.add(MaxPool2D(pool_size=(2, 2), padding='same'))
model.add(Flatten())
model.add(Dense(2, activation='softmax'))
model.summary()