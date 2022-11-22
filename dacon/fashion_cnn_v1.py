import sys
import os

import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns

from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.datasets import mnist
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D

from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from IPython.display import Image

for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Load the data

train = pd.read_csv("dacon/data/train.csv")
test = pd.read_csv("dacon/data/test.csv")

X_train = train.drop(["index","label"],axis=1)
Y_train = train['label']

test_index = test['index']
test = test.drop(['index'], 1)


X_train = X_train.values.reshape(-1,28,28,1)
test = test.values.reshape(-1,28,28,1)


class_names = ['T_shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

plt.figure(figsize=(20, 20))
for k in range(10):
  XX = X_train[Y_train == k]
  YY = Y_train[Y_train == k].reset_index()['label']
  for i in range(10):
    plt.subplot(10, 10, k*10 + i + 1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(XX[i][:,:,0], cmap='gray')
    label_index = int(YY[i])
    plt.title('{}. {}'.format(k, class_names[label_index]))
plt.show()


X_train, X_val, Y_train, Y_val = train_test_split(X_train, 
                                                  Y_train, 
                                                  test_size = 0.2, 
                                                  random_state=100)

print("Train set 이미지 수: {} 개".format(X_train.shape[0]))
print("Validation set 이미지 수: {} 개".format(X_val.shape[0]))

# CNN
X_train = X_train.reshape(X_train.shape[0], 28, 28, 1).astype('float32') / 255
X_val = X_val.reshape(X_val.shape[0], 28, 28, 1).astype('float32') / 255

Y_train = np_utils.to_categorical(Y_train)
Y_val = np_utils.to_categorical(Y_val)


# 컨볼루션 신경망 설정
model = Sequential()

model.add(Conv2D(32, kernel_size = (3,3), input_shape=(28,28,1), activation='relu'))
model.add(Conv2D(64,(3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=2))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))


model.summary()


model.compile(loss = 'categorical_crossentropy',
              optimizer='adam',
              metrics = ['accuracy'])


MODEL_DIR = "./model/"

if not os.path.exists(MODEL_DIR):
  os.mkdir(MODEL_DIR)

modelpath = './model/{epoch:02d}-{val_loss:.4f}.hdf5'
checkpointer = ModelCheckpoint(filepath=modelpath, monitor = 'val_loss', verbose=1, save_best_only=True)

early_stopping_callback = EarlyStopping(monitor='val_loss', patience=10)


history = model.fit(X_train, Y_train, validation_data = (X_val, Y_val), 
                    epochs=20, 
                    batch_size=200, 
                    verbose=0, 
                    callbacks=[early_stopping_callback, checkpointer])

print("\n Test Accuracy: %.4f" % (model.evaluate(X_val, Y_val)[1]))

fig, (ax1, ax2) = plt.subplots(1, 2, figsize = (12, 5))

# 오차
y_vloss = history.history['val_loss']

# 학습셋 오차
y_loss = history.history['loss']

# 그래프로 표현
x_len = np.arange(len(y_loss))
ax1.plot(x_len, y_vloss, marker = '.', c="red", label='Testset_loss')
ax1.plot(x_len, y_loss, marker = '.', c='blue', label = 'Trainset_loss')

# 그래프에 그리드를 주고 레이블을 표시
ax1.legend(loc='upper right')
ax1.grid()
ax1.set(xlabel='epoch', ylabel='loss')


# 정확도
y_vaccuracy = history.history['val_accuracy']

# 학습셋
y_accuracy = history.history['accuracy']

# 그래프로 표현
x_len = np.arange(len(y_accuracy))
ax2.plot(x_len, y_vaccuracy, marker = '.', c="red", label='Testset_accuracy')
ax2.plot(x_len, y_accuracy, marker = '.', c='blue', label = 'Trainset_accuracy')

# 그래프에 그리드를 주고 레이블을 표시
ax2.legend(loc='lower right')
ax2.grid()

ax2.set(xlabel='epoch', ylabel='accuracy')

# draw gridlines
ax2.grid(True)
plt.show()

# y_pred = model.predict(X_val).round(2)

# y_val_label = list(map(np.argmax, Y_val))
# y_pred_label = list(map(np.argmax, y_pred))

# plt.figure(figsize = (16,9))

# cm = confusion_matrix(y_val_label,y_pred_label)

# sns.heatmap(cm , annot = True,fmt = 'd',xticklabels = class_names,yticklabels = class_names)

# aaa = np.array(y_val_label) != np.array(y_pred_label)

# not_equel_list = np.where(aaa == True)[0]

# plt.figure(figsize=(20,20))
# j = 1
# for i in not_equel_list[0:36]:
# # for a in np.random.randint(0,206,36):
# #     i = not_equel_list[a]
# #     print(a)
#     plt.subplot(6,6,j); j+=1
#     plt.imshow(X_val[i].reshape(28,28),cmap = 'Greys')
#     plt.axis('off')
#     plt.title('Actual = {} / {} \nPredicted = {} / {}'.format(class_names[y_val_label[i]],
#                                                             y_val_label[i],
#                                                             class_names[y_pred_label[i]],
#                                                             y_pred_label[i]))


results = model.predict(test)
results = np.argmax(results,axis = 1)
results = pd.Series(results,name="label")



submission = pd.concat([pd.Series(range(1,28001),name = "index"),results],axis = 1)
submission.to_csv("results_fashion_mnist.csv",index=False)
