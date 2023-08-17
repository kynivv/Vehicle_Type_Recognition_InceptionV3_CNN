# Importing Libraries
import pandas as pd
from zipfile import ZipFile
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import layers
from sklearn.model_selection import train_test_split
import os
from glob import glob
import cv2
from keras.callbacks import ModelCheckpoint


# Extracting Zip Dataset
with ZipFile('vehicles.zip') as datazip:
    datazip.extractall('data')


# Hyperparameters
IMG_SIZE = 512
IMG_SHAPE = (IMG_SIZE, IMG_SIZE, 3)
BATCH_SIZE = 10
EPOCHS = 10
SPLIT = 0.2


# Data Preprocessing
data_path = 'data/Dataset'

classes = os.listdir(data_path)

X = []
Y = []

for i, name in enumerate(classes):
    images = glob(f'{data_path}/{name}/*.jpg')
    
    for image in images:
        img = cv2.imread(image)

        X.append(cv2.resize(img, (IMG_SIZE, IMG_SIZE)))
        Y.append(i)

X = np.asarray(X)
Y = pd.get_dummies(Y)


# Train Test Split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y,
                                                    test_size= SPLIT,
                                                    shuffle= True,
                                                    random_state= 42
                                                    )


# Creating Neural Network Model Based On InceptionV3
base_model = keras.applications.InceptionV3(include_top= False,
                                            pooling= 'max',
                                            weights= 'imagenet',
                                            input_shape= IMG_SHAPE
                                            )

model = keras.Sequential([
    base_model,
    layers.Dense(256, activation= 'relu'),
    layers.Dropout(0.25),
    layers.Dense(4, activation= 'softmax')
])

model.compile(optimizer= 'adam',
              loss= 'categorical_crossentropy',
              metrics= ['accuracy']
              )


# Creating Model Callbacks
model_checkpoint = ModelCheckpoint('output/model_checkpoint.h5',
                                   monitor= 'val_accuracy',
                                   verbose= 1,
                                   save_best_only= True,
                                   save_weights_only= True
                                   )


# Model Training
history = model.fit(X_train, Y_train,
                    batch_size= BATCH_SIZE,
                    epochs= EPOCHS,
                    validation_data= (X_test, Y_test),
                    shuffle= True,
                    callbacks= model_checkpoint,
                    verbose= 1,
                    )