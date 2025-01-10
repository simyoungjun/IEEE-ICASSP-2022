


import class_new


from class_new import *
import tensorflow as tf
import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
# import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, recall_score, accuracy_score, precision_score
from tensorflow.keras.models import load_model

known_volume_path = 'C:/Users/GJ/PycharmProjects/2022SPCUP/part2_code/part2_full_train_4X5000'
# eval_volume_path = '../spc


known_path, known_label_path = part2_file_path_list(known_volume_path, True)
# eval_path, eval_label_path = eval_file_path_list(eval_volume_path)


rs = 10

# pd_label =  pd.read_csv(known_label_path)
known_labels = known_label_path

knwon_train_path, known_test_path, y_train_raw, y_test_raw = train_test_split(np.array(known_path),
                                                                              known_labels, test_size=0.2,
                                                                              stratify=known_labels, random_state=rs)

n_mels = 64

train_data = data(knwon_train_path, known_label_path, n_mels=n_mels)
# eval_data = data(eval_path ,eval_label_path,n_mels=n_mels)

for data in [train_data]:
    data.extract_mel()
    data.reshape_data()

#     data.extract_cqt()
#     data.reshape_data_cqt_mel()

# X_train = train_data.X_reshaped
# X_eval = eval_data.X_reshaped


model = load_model('../10_part2_label6_spcup_save_last.h5')  # CNN_label_6_mel

model_vec = load_model('../10_part2_label6_spcup_save_last.h5')

model_vec.pop()
model_vec.compile()
# model_vec.summary()


from scipy.stats import mode

n_round = 5
# train_soft, train_pred = ext_soft(train_data, model, model_vec, n_round)

from sklearn.preprocessing import RobustScaler
train_soft, train_pred = ext_soft(train_data, model, model_vec, n_round)
scaler = RobustScaler()
train_sc = scaler.fit_transform(train_soft)

np.save('part2_train_sc_robust',train_sc)






known_volume_path = 'C:/Users/GJ/Desktop/연구실/2022SPCUP/spcup_2022_training_part1'
# eval_volume_path = 'C:/Users/GJ/PycharmProjects/2022SPCUP/spcup_2022_eval_part1'


known_path, known_label_path = eval_file_path_list(known_volume_path)
# eval_path, eval_label_path = eval_file_path_list(eval_volume_path)


rs = 10

# pd_label =  pd.read_csv(known_label_path)
known_labels = known_label_path

knwon_train_path, known_test_path, y_train_raw, y_test_raw = train_test_split(np.array(known_path),
                                                                              known_labels, test_size=0.2,
                                                                              stratify=known_labels, random_state=rs)

n_mels = 64

train_data = data(knwon_train_path, known_label_path, n_mels=n_mels)
# eval_data = data(eval_path ,eval_label_path,n_mels=n_mels)

for data in [train_data]:
    data.extract_mel()
    data.reshape_data()

#     data.extract_cqt()
#     data.reshape_data_cqt_mel()

# X_train = train_data.X_reshaped
# X_eval = eval_data.X_reshaped


model = load_model('../10_part2_label6_spcup_save_last.h5')  # CNN_label_6_mel

model_vec = load_model('../10_part2_label6_spcup_save_last.h5')

model_vec.pop()
model_vec.compile()
# model_vec.summary()


from scipy.stats import mode

n_round = 5
# train_soft, train_pred = ext_soft(train_data, model, model_vec, n_round)

from sklearn.preprocessing import RobustScaler
train_soft, train_pred = ext_soft(train_data, model, model_vec, n_round)
scaler = RobustScaler()
train_sc = scaler.fit_transform(train_soft)

np.save('part1_train_sc_robust',train_sc)