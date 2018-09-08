
from __future__ import print_function
from PIL import Image
import numpy as np
import shutil
import glob
import re
import cv2
import tensorflow as tf
import keras
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from keras import regularizers
from keras.layers import BatchNormalization, Dense, Dropout, Flatten, Conv2D, MaxPooling2D, GaussianNoise
from keras.models import load_model
from keras.callbacks import EarlyStopping, TensorBoard, ReduceLROnPlateau, ModelCheckpoint
from keras.models import Sequential
from keras import backend as K
from keras.layers.convolutional import ZeroPadding2D

gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.4) #40% of memory
sess_config = tf.ConfigProto(gpu_options=gpu_options)
sess = tf.Session(config=sess_config)
keras.backend.set_session(sess)

# read the file
# need to cd into the working direcotry
configfiles_train = glob.glob('Sorted_train/**/*.jpg', recursive=True)
configfiles_test = glob.glob('Sorted_test/**/*.jpg', recursive=True)
configfiles_new = glob.glob('new_data/**/*.jpg', recursive=True)

# get the dimension
im = Image.open(configfiles_train[1])
height, width = np.shape(im)
new_height, new_width = 100, 218
new_height_2, new_width_2 = 90, 135

# preprocess the image
x = []
y = []
for img in configfiles_train:
    m = re.search('Sorted_train\/([0-9]+)\/.+', img)
    img = cv2.imread(img, 0)
    img = cv2.resize(img, (new_width, new_height))
    img = img[5:95, 33:168]
    equ = cv2.equalizeHist(img)
    # norm = np.array([])
    # equ = cv2.normalize(equ, norm, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    if equ.shape != [new_height_2, new_width_2, 1]:
        equ = equ.reshape(new_width_2, new_height_2, 1)
    x.append(equ)
    y.append(m.group(1))

for img in configfiles_test:
    m = re.search('Sorted_test\/([0-9]+)\/.+', img)
    img = cv2.imread(img, 0)
    img = cv2.resize(img, (new_width, new_height))
    img = img[5:95, 33:168]
    equ = cv2.equalizeHist(img)
    if equ.shape != [new_height_2, new_width_2, 1]:
        equ = equ.reshape(new_width_2, new_height_2, 1)
    x.append(equ)
    y.append(m.group(1))

for img in configfiles_new:
    m = re.search('new_data\/([0-9]+)\/.+', img)
    img = cv2.imread(img, 0)
    img = cv2.resize(img, (new_width, new_height))
    img = img[5:95, 33:168]
    equ = cv2.equalizeHist(img)
    if equ.shape != [new_height_2, new_width_2, 1]:
        equ = equ.reshape(new_width_2, new_height_2, 1)
    x.append(equ)
    y.append(m.group(1))

# split the data
x = np.array(x)
y = np.array(y)
(x_train, x_test, y_train, y_test) = train_test_split(x, y, test_size = 0.3, random_state=42)
(x_valid, x_test, y_valid, y_test) = train_test_split(x_test, y_test, test_size=0.5, random_state=42)

input_shape = (new_width_2, new_height_2, 1)
batch_size = 138
num_classes = 131
epochs = 100
dropout_p = 0.25
use_batchnorm = True

x_train = x_train.astype('float32')
x_valid = x_valid.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_valid /= 255
x_test /= 255

# turn the labels into categories
y_train = y_train.astype(int)
y_valid = y_valid.astype(int)
y_test = y_test.astype(int)
y_tr = keras.utils.to_categorical(y_train, num_classes)
y_va = keras.utils.to_categorical(y_valid, num_classes)
y_te = keras.utils.to_categorical(y_test, num_classes)

'''
from keras.preprocessing.image import ImageDataGenerator
datagen = ImageDataGenerator(
                             featurewise_center=True,
                             featurewise_std_normalization=False,
                             rotation_range=5,
                             width_shift_range=0.1,
                             height_shift_range=0.1,
                             shear_range=0.01,
                             horizontal_flip=False)
datagen.fit(x_train)'''

# build the model
model = Sequential()
model.add(Conv2D(filters=16, kernel_size=(3, 3), activation='selu', input_shape=input_shape, kernel_regularizer=regularizers.l2(0.01), bias_regularizer=regularizers.l1(0.01)))
model.add(ZeroPadding2D(padding=(1,1)))
model.add(MaxPooling2D(pool_size=(2, 2), strides = 2))
model.add(Conv2D(filters=16, kernel_size=(2, 2), activation='selu', input_shape=input_shape, kernel_regularizer=regularizers.l2(0.01), bias_regularizer=regularizers.l1(0.01)))
model.add(ZeroPadding2D(padding=(1,1)))
model.add(MaxPooling2D(pool_size=(2, 2), strides = 2))
if use_batchnorm:
    model.add(BatchNormalization())
model.add(GaussianNoise(0.1))
model.add(Dropout(dropout_p))

model.add(Conv2D(filters=32, kernel_size=(3, 3), activation='selu', kernel_regularizer=regularizers.l2(0.01), bias_regularizer=regularizers.l1(0.01)))
model.add(ZeroPadding2D(padding=(1,1)))
model.add(MaxPooling2D(pool_size=(2, 2), strides = 2))
model.add(Conv2D(filters=32, kernel_size=(2, 2), activation='selu', kernel_regularizer=regularizers.l2(0.01), bias_regularizer=regularizers.l1(0.01)))
model.add(ZeroPadding2D(padding=(1,1)))
model.add(MaxPooling2D(pool_size=(2, 2), strides = 2))
if use_batchnorm:
    model.add(BatchNormalization())
model.add(GaussianNoise(0.1))
model.add(Dropout(dropout_p))

model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='selu', kernel_regularizer=regularizers.l2(0.01), bias_regularizer=regularizers.l1(0.01)))
model.add(ZeroPadding2D(padding=(1,1)))
model.add(MaxPooling2D(pool_size=(2, 2), strides = 2))
model.add(Conv2D(filters=64, kernel_size=(2, 2), activation='selu', kernel_regularizer=regularizers.l2(0.01), bias_regularizer=regularizers.l1(0.01)))
model.add(ZeroPadding2D(padding=(1,1)))
model.add(MaxPooling2D(pool_size=(2, 2), strides = 2))
if use_batchnorm:
    model.add(BatchNormalization())
model.add(GaussianNoise(0.1))
model.add(Dropout(dropout_p))

model.add(Flatten())
model.add(Dense(256))
if use_batchnorm:
    model.add(BatchNormalization())
model.add(Dropout(0.5))
model.add(Dense(num_classes,
                  activation='softmax'))
model.compile(loss='categorical_crossentropy',
                optimizer='adam',
                metrics=['accuracy'])

# callbacks setup
early_stopping_monitor = EarlyStopping(monitor='val_loss',
                                       patience = 10, verbose=1,
                                       mode='auto')
tensorboard = TensorBoard(log_dir='./logs')
reduceLR = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=8, verbose=1, mode='auto',min_delta=0.0001, cooldown=0, min_lr=0.000001)
checkpointer = ModelCheckpoint(filepath='weights.{epoch:02d}-{val_loss:.2f}-{val_acc:.4f}.hdf5', monitor='val_acc', verbose=1, save_best_only=True)

'''model_2.fit_generator(datagen.flow(x_train, y_tr,
            batch_size=batch_size),
            epochs=epochs,
            verbose=1,
            validation_data=(x_test, y_te),
            callbacks = [tensorboard, reduce_lr])'''

model.fit(x_train, y_tr, batch_size=batch_size,
                  epochs=epochs,
                  verbose=1,
                  validation_data=(x_valid, y_va),
                  callbacks = [early_stopping_monitor,tensorboard, reduceLR, checkpointer])

# load model
model = load_model('weights.83-0.11-0.9936.hdf5')
score = model.evaluate(x_test, y_te, verbose = 1)
print("\nScore is: {}".format(score))

# Create the plot
import matplotlib.pyplot as plt
plt.plot(model_training.history['val_loss'], 'b')
plt.xlabel('Epochs')
plt.ylabel('Validation Loss')
plt.savefig("Trend.png")
plt.close()

# predict on all images
x = x.astype('float32')
x /= 255
y = y.astype(int)

y_pred = model.predict(x, verbose = 1)
y_pred_label = y_pred.argmax(axis = -1)

# storing mislabelled images
incorrect = []
for i in range(len(y)):
	if (y_pred_label[i] != y[i]):
		incorrect.append(i)
total_pictures = configfiles_train + configfiles_test + configfiles_new

# storing images into folders
import os
os.mkdir('misclassified_image')
path = 'misclassified_image'

for index in incorrect:
	img_name = total_pictures[index]
	img = cv2.imread(img_name, 0)
	file_name = str(y_pred_label[index]) + '.jpg'
	cv2.imwrite(os.path.join(path, file_name), img)

# get the filenames of misclassified images
image_names = []
for index in incorrect:
	image_names.append(total_pictures[index])
