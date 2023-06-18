import numpy as np
from matplotlib import pyplot
from tensorflow import keras
from keras.models import Model, Sequential
from keras.layers import Dense, Dropout, Flatten, Activation
from keras.layers import Conv2D, MaxPooling2D,GlobalAveragePooling2D
from keras import regularizers, optimizers
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import SGD, Adam
from keras.layers.normalization import BatchNormalization
from keras.utils import np_utils
from keras.callbacks import TensorBoard,ReduceLROnPlateau,ModelCheckpoint
from tensorflow.keras.applications import ResNet50

batch_size = 20
train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    '/content/drive/My Drive/DATASET/TRAINING_SET',  # this is the target directory
    target_size=(150, 150),  # all images will be resized to 150x150
    batch_size=batch_size,
    class_mode='categorical')  

validation_generator = test_datagen.flow_from_directory(
    '/content/drive/My Drive/DATASET/TESTING_SET',
    target_size=(150, 150),
    batch_size=batch_size,
    class_mode='categorical')

img_height,img_width = 150,150
num_classes = 95

base_model = ResNet50(weights= None, include_top=False, input_shape= (img_height,img_width,3))

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.7)(x)
predictions = Dense(num_classes, activation= 'softmax')(x)
model = Model(inputs = base_model.input, outputs = predictions)

adam = Adam(lr=0.00001)
model.compile(optimizer= adam, loss='categorical_crossentropy', metrics=['accuracy'])

history = model.fit(train_generator,epochs=50,validation_data=validation_generator)
