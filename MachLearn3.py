import numpy as np
import tensorflow as tf
import os 
from random import shuffle 
from keras.models import Model
from keras.layers import Input, Activation, Dense, Conv2D, MaxPooling2D, ZeroPadding2D, Flatten
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from keras.callbacks import TensorBoard
import time

NAME = "Mach-Learning-{}".format(int(time.time()))
callbacks = TensorBoard(log_dir='./Graph/{}'.format(NAME))

dirTrain = 'TrainData'
dirValidation = 'ValidationData'
TrainingData = 393+363
ValidationData = 33*2
batch_size = 32
img_size = 224
epochs = 20

train_datagen = ImageDataGenerator(
    rescale=1. /255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

validation_gen = ImageDataGenerator(
    rescale=1. /255
    )

train_generator = train_datagen.flow_from_directory(
    dirTrain,
    target_size=(img_size,img_size),
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=True,
    color_mode='rgb' 
)

validation_generator = validation_gen.flow_from_directory(
    dirValidation,
    target_size=(img_size,img_size),
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=True,
    color_mode='rgb' 
)

I_layer = Input(shape=(img_size,img_size,3))
C_layer = ZeroPadding2D(padding=(2,2))(I_layer)
C_layer = Conv2D(16, (5,5), strides=(2,2), activation='relu' )(C_layer)
C_layer = MaxPooling2D((2,2))(C_layer)
C_layer = Conv2D(32, (5,5), strides=(1,1), activation='relu')(C_layer)
C_layer = Conv2D(32, (5,5), strides=(1,1), activation='relu')(C_layer)
C_layer = MaxPooling2D((2,2))(C_layer)
C_layer = Conv2D(64, (3,3), strides=(1,1), activation='relu')(C_layer)

F_layer = Flatten()(C_layer)

FC_layer = Dense(256, activation='relu')(F_layer)
FC_layer = Dense(64, activation='relu')(FC_layer)
FC_layer = Dense(32, activation='relu')(FC_layer)
FC_layer = Dense(8, activation='relu')(FC_layer)
O_layer = Dense(2, activation='sigmoid')(FC_layer)

Ress_model = Model(inputs=I_layer, outputs=O_layer)

adam = Adam(lr=0.0001)
Ress_model.compile(optimizer=adam, loss='binary_crossentropy', metrics=['accuracy'])

print(Ress_model.summary())

Ress_model.fit_generator(
    train_generator,
    steps_per_epoch=TrainingData // batch_size,
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps=ValidationData // batch_size, 
    callbacks=[callbacks]
)

Ress_model.save_weights('MachLearn3out1.h5')
Ress_model.save('MachLearn3Out1.model')