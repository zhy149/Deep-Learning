import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.metrics import mean_squared_error

data = np.load('Robot_Trials_DL.npy')
np.random.shuffle(data)
train_set = data[0:413]  # training set
valid_set = data[413:551]  # valid set
test_set = data[551:]  # test set
np.save('testset.npy', test_set)  # print(np.transpose(train_set)[0][:,2].shape)
grd_truth_train = train_set[:, :, 1]  # ground truth for train set, water volume
train_set = np.delete(train_set, 1, axis=2)  # new train set without output
train_set = np.transpose(train_set, (2, 0, 1))  # turn to (6, 413, 700)
grd_truth_valid = valid_set[:, :, 1]  # ground truth for valid set, water volume
valid_set = np.delete(valid_set, 1, axis=2)
valid_set = np.transpose(valid_set, (2, 0, 1))  # turn to (6, 138, 700)
stop_index_train = []
stop_index_valid = []
for i in range(len(grd_truth_train)):  # calculate train set stop indexes
    for j in reversed(range(700)):
        if grd_truth_train[i][j] != 0:
            break
    stop_index_train.append(j)
for i in range(len(grd_truth_valid)):  # calculate valid set stop indexes
    for j in reversed(range(700)):
        if grd_truth_valid[i][j] != 0:
            break
    stop_index_valid.append(j)
mean_dev = []  # store mean and dev for 6 features, same for all sets
for i in range(6):  # normalize each feature
    tmp_lis = []
    for j in range(len(train_set[i])):
        tmp_lis.extend(train_set[i][j][:stop_index_train[j]])
    temp_mean = np.mean(tmp_lis)
    temp_std = np.std(tmp_lis)
    mean_dev.append((temp_mean, temp_std))
    for j in range(len(train_set[i])):  # normalize the ith feature in jth sequence
        train_set[i][j][:stop_index_train[j]] -= temp_mean
        train_set[i][j][:stop_index_train[j]] /= temp_std
    for j in range(len(valid_set[i])):
        valid_set[i][j][:stop_index_valid[j]] -= temp_mean
        valid_set[i][j][:stop_index_valid[j]] /= temp_std
# finish normalization
train_set = np.transpose(train_set, (1, 2, 0))  # turn to (413, 700, 6)
valid_set = np.transpose(valid_set, (1, 2, 0))  # turn to (138, 700, 6)
grd_truth_train = grd_truth_train[:, :, np.newaxis]
grd_truth_valid = grd_truth_valid[:, :, np.newaxis]

my_model = keras.Sequential()
my_model.add(layers.Input(shape=(train_set.shape[1], 6)))
my_model.add(layers.Masking())
my_model.add(layers.LSTM(16, return_sequences=True))
my_model.add(layers.LSTM(16, return_sequences=True))
my_model.add(layers.LSTM(16, return_sequences=True))
my_model.add(layers.LSTM(16, return_sequences=True))
my_model.add(layers.Dropout(0.2))
my_model.add(layers.Dense(units=8, activation="relu"))
my_model.add(layers.Dense(1))

my_opt = keras.optimizers.Adam(learning_rate=0.001)
my_model.compile(loss=keras.losses.mean_squared_error, optimizer=my_opt)
my_model.summary()
my_model.fit(x=train_set, y=grd_truth_train, batch_size=25, epochs=200, validation_data=(valid_set, grd_truth_valid),
             verbose=1)  # callbacks = [rlrp]?
my_model.save('model_one_new')