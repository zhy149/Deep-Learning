import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.metrics import mean_squared_error
import sys

test_file = sys.argv[1]  # test file name
mean_dev = [(-44.09636025116822, 36.16324315995359), (0.7168738024483643, 0.2534629105845952), (0.2453317578031854, 0.12249926507499104), (225.5226305230709, 83.61513361951928), (86.68012809011977, 16.23552901267466), (0.08025621489065836, 0.5180108012382653)]
test_set = np.load(test_file)
grd_truth_test = test_set[:, :, 1]  # ground truth for test set, water volume
test_set = np.delete(test_set, 1, axis=2)  # new test set without output
test_set = np.transpose(test_set, (2, 0, 1))  # turn to (6, sample amount, 700)
stop_index_test = []
for i in range(len(grd_truth_test)):  # calculate test set stop indexes
    for j in reversed(range(700)):
        if grd_truth_test[i][j] != 0:
            break
    stop_index_test.append(j)

for i in range(6):
    tmp_mean, tmp_std = mean_dev[i]
    for j in range(len(test_set[i])):
        test_set[i][j][:stop_index_test[j]] -= tmp_mean
        test_set[i][j][:stop_index_test[j]] /= tmp_std
test_set = np.transpose(test_set, (1, 2, 0))  # turn to (413, 700, 6)
grd_truth_test = grd_truth_test[:, :, np.newaxis]
trained_model = keras.models.load_model('model_new')
output_after = trained_model(test_set)
write_back_array = output_after.numpy()
for i in range(len(stop_index_test)):
    for j in range(stop_index_test[i]+1, 700):
        write_back_array[i][j] = 0  # padding zeros
np.save('test_set_output.npy', write_back_array)  # print(np.transpose(train_set)[0][:,2].shape)
cal_error = mean_squared_error(grd_truth_test[:, :, 0], write_back_array[:, :, 0])
print("loss on test set: ", cal_error)
