# -*- coding: utf-8 -*-
"""
Created on Thu Jul 30 15:58:38 2020

@author: 91989
"""
from keras.preprocessing.image import ImageDataGenerator
data_generator = ImageDataGenerator(
                        featurewise_center=False,
                        featurewise_std_normalization=False,
                        rotation_range=10,
                        width_shift_range=0.1,
                        height_shift_range=0.1,
                        zoom_range=.1,
                        horizontal_flip=True)

classifier = Sequential()


classifier.add(Convolution2D(32, 3, 3, input_shape = (48, 48, 1), activation = 'relu'))
classifier.add(BatchNormalization())
classifier.add(MaxPooling2D(pool_size = (2, 2)))
classifier.add(Dropout(0.5))

classifier.add(Convolution2D(32, 3, 3, activation = 'relu'))
classifier.add(BatchNormalization())
classifier.add(MaxPooling2D(pool_size = (2, 2)))
classifier.add(Dropout(0.5))

classifier.add(Convolution2D(32, 3, 3, activation = 'relu'))
classifier.add(BatchNormalization())
classifier.add(MaxPooling2D(pool_size = (2, 2)))
classifier.add(Dropout(0.5))


classifier.add(Flatten())

classifier.add(Dense(output_dim = 128, activation = 'relu'))
classifier.add(Dense(output_dim = 7, activation = 'softmax'))


classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

classifier.fit_generator(data_generator.flow(xtrain, ytrain,32),
                         samples_per_epoch = len(xtrain)/32,
                         nb_epoch = 25,
                         validation_data = (xtest,ytest))