from keras.models import Sequential
from keras.optimizers import Adam
from keras.layers.core import Dense, Activation, Dropout, Flatten
from keras.layers import CuDNNLSTM
from keras.layers.recurrent import LSTM
from keras.utils import np_utils

import numpy as np
import csv
import time
import matplotlib.pylab as plt

print('hello!!!')

current = []
charge_rate = []
z = []
np.set_printoptions(threshold=np.nan)

time_steps = 20
no_classes = 16
no_features = 2

with open('data/charging_data.csv') as csvfile:
    reader = csv.reader(csvfile)
    reader.__next__()
    for row in reader:
        current.append(float(row[0]))
        charge_rate.append(int(row[1]))
        z.append(float(row[2]))

x_all = np.empty([len(current) - time_steps + 1, time_steps, no_features])
y_all = np.empty([len(current) - time_steps + 1])

for i in range(len(current) - time_steps + 1):
    x_all[i, :, 0] = current[i:i + time_steps]
    x_all[i, :, 1] = z[i: i + time_steps]
    y_all[i] = charge_rate[i + time_steps - 1]

    # print(i, x_all[i, :, 0], x_all[i, :, 1], y_all[i])

print('all shape: ', x_all.shape, y_all.shape)
# x_train = x_all[0:4600, :, :]
# x_test = x_all[4600:, :, :]
#
# y_train = y_all[0:4600]
# y_test = y_all[4600:]
#
# print(x_train.shape, x_test.shape)
# print(y_train.shape, y_test.shape)
#
# y_train = np_utils.to_categorical(y_train, no_classes)
# y_test = np_utils.to_categorical(y_test, no_classes)
##################################################################
current[:] = []
charge_rate[:] = []
z[:] = []
with open('data/charging_data1.csv') as csvfile:
    reader = csv.reader(csvfile)
    reader.__next__()
    for row in reader:
        current.append(float(row[0]))
        charge_rate.append(int(row[1]))
        z.append(float(row[2]))

x_temp = np.empty([len(current) - time_steps + 1, time_steps, no_features])
y_temp = np.empty([len(current) - time_steps + 1])

for i in range(len(current) - time_steps + 1):
    x_temp[i, :, 0] = current[i:i + time_steps]
    x_temp[i, :, 1] = z[i: i + time_steps]
    y_temp[i] = charge_rate[i + time_steps - 1]

print('2 shape: ', x_temp.shape, y_temp.shape)
print('1 shape: ', x_all.shape, y_all.shape)

x_all = np.vstack((x_all, x_temp))
y_all = np.concatenate([y_all, y_temp])

print(x_all.shape, y_all.shape)

training_portion = int(x_all.shape[0] * 0.8)

x_train = x_all[0:training_portion - 1, :, :]
x_test = x_all[training_portion:, :, :]

y_train = y_all[0:training_portion - 1]
y_test = y_all[training_portion:]

print(x_train.shape, x_test.shape)
print(y_train.shape, y_test.shape)

y_train = np_utils.to_categorical(y_train, no_classes)
y_test = np_utils.to_categorical(y_test, no_classes)

model = Sequential()

model.add(
    CuDNNLSTM(64, stateful=True, batch_input_shape=[1, x_train.shape[1], x_train.shape[2]]))
##model.add(CuDNNLSTM(64, return_sequences=False))

##model.add(LSTM(64, return_sequences=True, input_shape=(x_train.shape[1:])))
# model.add(LSTM(128, return_sequences=True, input_shape=(x_train.shape[1:])))
# model.add(LSTM(64, return_sequences=True))
# model.add(LSTM(64, return_sequences=False))
# model.add(Dropout(0.5))
model.add(Dense(no_classes, activation='sigmoid'))

adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.00)
model.compile(loss='categorical_crossentropy',
              optimizer=adam,
              metrics=['accuracy'])

for i in range(1):
    model.fit(x_train, y_train, batch_size=1, epochs=1, verbose=1, shuffle=False)
    model.reset_states()
score = model.evaluate(x_test, y_test, batch_size=128)
print("Model Accuracy: %.2f%%" % (score[1]*100))
# print(score)
# plt.plot(history.history['acc'])
# plt.plot(history.history['val_acc'])
# plt.title('model accuracy')
# plt.ylabel('accuracy')
# plt.xlabel('epoch')
# plt.legend(['train', 'test'], loc='upper left')
# plt.show()
# # summarize history for loss
# plt.plot(history.history['loss'])
# plt.plot(history.history['val_loss'])
# plt.title('model loss')
# plt.ylabel('loss')
# plt.xlabel('epoch')
# plt.legend(['train', 'test'], loc='upper left')
# plt.show()
#
# model.save('model8.h5')
