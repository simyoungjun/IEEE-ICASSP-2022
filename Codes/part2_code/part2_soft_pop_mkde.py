#!/usr/bin/env python
# coding: utf-8

# In[ ]:

import sys
sys.path.append('C:/Users/GJ/PycharmProjects/2022SPCUP')

import class_new
import importlib
importlib.reload(class_new)


from class_new import *
import tensorflow as tf
import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, recall_score, accuracy_score, precision_score



known_volume_path = './part2_full_train_4X5000'
unknown_volume_path = './unseen_noisy_full'



rs = 42
known_path, known_labels = part2_file_path_list(known_volume_path,True)
unknown_path, unknown_labels = part2_file_path_list(unknown_volume_path,False)
##train set
# print('raw train_set_num :',len(labels))
X_train_path, X_test_path, y_train_raw, y_test_raw = train_test_split(np.array(known_path),
                                                                      known_labels, test_size=0.2,
                                                                      stratify = known_labels, random_state=rs)

n = 100
n_mels = 64
train = data(X_train_path,y_train_raw,n_mels=n_mels, known = True)

test = data(X_test_path,y_test_raw, n_mels=n_mels, known = True)

unseen =data(unknown_path,unknown_labels, n_mels=n_mels, known = False)

# train = data(X_train_path[:n],y_train_raw[:n],n_mels=n_mels, known = True)

# test = data(X_test_path[:n],y_test_raw[:n], n_mels=n_mels, known = True)

# unseen = data(unknown_path[:n],unknown_labels[:n], n_mels=n_mels, known = False)

for i in [ unseen, train, test]:
    i.extract_mel()
    i.reshape_data()


from scipy.stats import mode

prediction = model.predict(test.X_reshaped)
predicted_classes = np.argmax(prediction, axis=1)
f = 0
test_predict = []
softmax_val = []

for j,i in enumerate(test.file_split_num):
#     print(y_test_raw[j])
#     print(np.round(prediction[f:f + i],3))
#     print(predicted_classes[f:f + i])
#     print('\n')
    softmax_val.append(prediction[f:f + i])
    test_predict.append(mode(predicted_classes[f:f + i])[0][0])
    f += i
    
from sklearn.metrics import confusion_matrix
cf = confusion_matrix(y_test_raw, test_predict)
print('\n\n', cf)


# In[ ]:


model = load_model('./models_log/11_spcup_best_weight.h5')
# model = load_model('./models_log/40_42_spcup_best_model_act_split.h5')

model_vec = load_model('./models_log/11_spcup_best_weight.h5')

model_vec.pop()
model_vec.compile()
model_vec.summary()

def ext_soft(data,model, model_vec, n_round = 5):
    

    
    prediction_soft = model.predict(data.X_reshaped)
    prediction_vec = model_vec.predict(data.X_reshaped)
    predicted_classes = np.argmax(prediction_soft, axis=1)
    

    
    f = 0
    softmax_val = []
    test_predict = []

    for j,i in enumerate(data.file_split_num):
        mean_vec = np.mean(prediction_vec[f:f + i],axis = 0)    
        mean_soft = np.mean(prediction_soft[f:f + i],axis = 0) 
#         mean_soft = np.mean(prediction[f:f + i],axis = 0)
#         mean_soft = np.mean(np.power(prediction[f:f + i],2),axis = 0)
        
        softmax_val.append(np.round(mean_vec,n_round))
        test_predict.append(mode(predicted_classes[f:f + i])[0][0])
        
#         predicted_classes = np.argmax(np.mean(np.power(prediction[f:f + i],1),axis = 0))
#         test_predict.append(predicted_classes)
        f += i
    return np.array(softmax_val), np.array(test_predict)



test_soft, test_predict = ext_soft(test,model, model_vec)
unseen_soft, unseen_predict = ext_soft(unseen, model, model_vec)
train_soft, train_predict = ext_soft(train, model, model_vec)
# split_soft = model.predict(train.X_reshaped, model, model_vec)

#round 5로 했을 때 최고

scaler = StandardScaler()
test_sc = scaler.fit_transform(test_soft)
unseen_sc = scaler.fit_transform(unseen_soft)
train_sc = scaler.fit_transform(train_soft)
# split_sc = scaler.fit_transform(split_soft)


print(test_soft.shape)
print(unseen_soft.shape)
print(train_soft.shape)
# print(split_soft.shape)

print(test_sc[:3])
print(train_sc[:3])
print(unseen_sc[:3])


# In[ ]:


import time
def make_pdf_eval(data, pa):
    var_type = 'o' * data.shape[1]

    std_feature = np.std(data, axis=0)
    print(data.shape)
    d = data.shape[1]
    feature_length = data.shape[0]
    c = (4 / (d + 2) / feature_length) ** (1 / (d + 4))
    bw = std_feature * c
    print(bw)
#     print(bw.shape)
    # bw = bw.transpose()

    bw = bw * pa
    dens = sm.nonparametric.KDEMultivariate(data=data, var_type=var_type, bw=bw * pa)
    return dens


for pa in np.arange(0.01,0.7,0.01):
# for pa in [2.6000000000000014]:
    print(pa)
#     dens = make_pdf_eval(split_sc, pa)
    dens = make_pdf_eval(train_sc, pa)
    
    all_time = time.time()
    pdf_t = dens.pdf(test_sc)
    pdf_u = dens.pdf(unseen_sc)
#     pdf_tr = dens.pdf(split_sc)
    pdf_tr = dens.pdf(train_sc)
    

    pdf_all =  np.concatenate((pdf_t,pdf_u),axis = 0)
    print('all time : ', time.time() -all_time)
    
    print(max(pdf_tr))
    print(max(pdf_t))
    pdf_all = pdf_all/max(pdf_tr)
    pdf_at = pdf_all[:pdf_t.size]
    pdf_au = pdf_all[pdf_t.size:]
#     print(len(pdf_at[pdf_at == min(pdf_all)]))
    
    plt.plot(pdf_all)
    plt.ylim([np.min(pdf_all),np.sort(pdf_at)[-1]])
#     plt.ylim([np.min(pdf_all),np.max(pdf_)[-8]])
    
    plt.show()
    for i in np.unique(pdf_at)[:50]:
        print(i)
        print(len(pdf_at[pdf_at >= i]))
        print(len(pdf_au[pdf_au >= i]))
        print(len(pdf_at[pdf_at < i])+len(pdf_au[pdf_au < i]))
        
    print('----------------------------------')

