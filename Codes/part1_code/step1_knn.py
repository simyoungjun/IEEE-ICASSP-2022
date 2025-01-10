
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
from sklearn.neighbors import KNeighborsClassifier


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

train_kn = min_max_scale(train.flatten())
test_kn = min_max_scale(test.flatten())
unseen_kn = min_max_scale(unknown.flatten())

for i in range(3,7):
    neigh = KNeighborsClassifier(n_neighbors=i)
    neigh.fit(train_kn, train.labels_flatten)

