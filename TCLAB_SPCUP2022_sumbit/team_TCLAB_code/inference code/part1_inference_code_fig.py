# %
import submit_classes
import importlib

importlib.reload(submit_classes)

from submit_classes import *
# import tensorflow as tf
# import pandas as pd
# import numpy as np
# import tensorflow as tf
# import matplotlib.pyplot as plt
# import seaborn as sns
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler
# from sklearn.metrics import confusion_matrix, recall_score, accuracy_score, precision_score

# %
####load data and feature extraction
eval_volume_path = 'C:/Users/sim/연구실/음성합성분류 전자공학회/spcup_2022_eval_part1'

eval_path, eval_label_path = eval_file_path_list(eval_volume_path)



n_mels = 64

eval_data = data(eval_path, eval_label_path, n_mels=n_mels)

for data in [eval_data]:
    data.extract_mel()
    data.reshape_data()

X_eval = eval_data.X_reshaped

########load model and make logit, softmax
train_sc = np.load("./part1_train_sc_robust.npy")

print(train_sc.shape)

model = load_model('./10_part2_label6_spcup_save_last.h5')
model.summary()

model_vec = load_model('./10_part2_label6_spcup_save_last.h5')

model_vec.pop()
model_vec.compile()


# %
n_round = 5
eval_soft, eval_pred = ext_soft(eval_data, model, model_vec, n_round)

for k in range(6):
    print(len(eval_pred[eval_pred == k]))

scaler = RobustScaler()

eval_sc = scaler.fit_transform(eval_soft)

print(train_sc[:3])
print(eval_sc[:3])

pa = 0.03
dens = make_pdf(train_sc, pa)

pdf_tr = dens.pdf(train_sc)
pdf = dens.pdf(eval_sc)

pdf = pdf / max(pdf_tr)

for k in range(6):
    print(len(eval_pred[eval_pred == k]))

print('\n')

import copy

mkde_eval_pred = copy.copy(eval_pred)

thresholds = [0.5e-09,0.5e-07,0.5e-07, 0.5e-07 ,2e-06]
for i, thresh in enumerate(thresholds):
    idx_label = np.where(mkde_eval_pred == i)[0]
    idx_thresh = np.where(pdf[idx_label] < thresh)[0]
    idx = idx_label[idx_thresh]
    mkde_eval_pred[idx] = 5

print('')
print('mkde o :', end='  ')
for k in range(6):
    print(len(mkde_eval_pred[mkde_eval_pred == k]), end='  ')











# %
file_name = "./answer_part1.txt"
with open(file_name, "w") as file:
    for dirName, subdirList, fileList in os.walk(eval_volume_path):
        for i, filename in enumerate(fileList):
            if i != 9000:
                file.write(str(filename))
                file.write(', ')
                #                 file.write(str(eval_pred[i]))
                file.write(str(mkde_eval_pred[i]))

                file.write('\n')
            else:
                file.close()


filename_list = []
test_predict_value = []
for dirName, subdirList, fileList in os.walk(eval_volume_path):
    for i, filename in enumerate(fileList):
        if i != 9000:
            filename_list.append(filename)
            test_predict_value.append(mkde_eval_pred[i])
            dic = dict(track=filename_list, algorithm=test_predict_value)

df = pd.DataFrame(dic)
new_csv_file = df.to_csv(r'answer_part1.csv', index=False)