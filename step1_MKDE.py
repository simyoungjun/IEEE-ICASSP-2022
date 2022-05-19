# import import_ipynb
from classes import *
import classes
# importlib.reload(classes)
from sklearn.preprocessing import StandardScaler,MinMaxScaler

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

n = 50
n_mels = 32
train = classes.data(X_train_path[:n],y_train_raw[:n],n_mels=n_mels, known = True)
# X_train,y_train = train.extract_mel(sampling_rate,n_mels)

test = classes.data(X_test_path[:n],y_test_raw[:n], n_mels=n_mels, known = True)
# X_test,y_test = test.extract_mel(sampling_rate,n_mels)

unknown =data(unknown_path[:n],unknown_labels[:n], n_mels=n_mels, known = False)
# X_unknown,y_unknown = unknown.extract_mel(sampling_rate,n_mels)



scaler = MinMaxScaler()
for pa in [0.1,0.5,1,1.5]:
    print('\n\n--------------pa : ', pa)
    known_pd = mkde_test(n,n, train, test, n_mels, pa)
    unknown_pd = mkde_test(n, n, train, unknown, n_mels, pa)
    # known_pd =scaler.fit_transform(known_pd)
    # unknown_pd =scaler.fit_transform(unknown_pd)

    all_pd = np.concatenate((known_pd,unknown_pd),axis = None)
    all_pd.shape
    plt.subplot()
    plt.plot(all_pd)
    plt.show(block = False)
    plt.pause(1)
    plt.close()
    print('max_all_pd', max(all_pd))
    for thresh in np.arange(0,1,0.1):
        print('threshold :', thresh)
        mkde_file_acc(known_pd/max(all_pd), test,thresh)
        mkde_file_acc(unknown_pd/max(all_pd), unknown,thresh)




#
#
# # predecttion = density.dens.pdf(unknown_d.reshaped_data)
# #
# # test_d = classes.mkde(test.X_split)
# # prediction_test = density.dens.pdf(unknown_d.reshaped_data)