import keras
import os
import shutil
from keras.models import *
from keras.layers import *
from keras.optimizers import *
import numpy as np
import cv2 as cv
from keras.preprocessing.image import ImageDataGenerator

batch_size = 100
# this is the augmentation configuration we will use for training
train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

# this is the augmentation configuration we will use for testing:
# only rescaling
test_datagen = ImageDataGenerator(rescale=1./255)

# this is a generator that will read pictures found in
# subfolers of 'data/train', and indefinitely generate
# batches of augmented image data
train_generator = train_datagen.flow_from_directory(
        'data/train',  # this is the target directory
        target_size=(60, 60),  # all images will be resized to 150x150
        batch_size=batch_size,
        class_mode='binary')  # since we use binary_crossentropy loss, we need binary labels

# this is a similar generator, for validation data
validation_generator = test_datagen.flow_from_directory(
        'data/validation',
        target_size=(60, 60),
        batch_size=batch_size,
        class_mode='binary')

model = Sequential()


# input shape 24 x 60 x 60 x 3		
model.add(Conv2D(8, kernel_size=(3, 3),
				 activation='relu',
				 input_shape=(60,60,3)))
model.add(Conv2D(16, kernel_size=(3, 3),
				 activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(24, kernel_size=(3, 3),
				 activation='sigmoid'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))

model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.4))
model.add(Dense(1, activation='softmax'))

model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['accuracy'])

model.fit_generator(
        train_generator,
        steps_per_epoch=100,
        epochs=2000,
        validation_data=validation_generator,
        validation_steps=50)

path = "save_data"
if os.path.exists(path):
	shutil.rmtree(path)
os.mkdir(path)	
model_json = model.to_json()
with open(path+"/model.json", "w") as json_file:
	json_file.write(model_json)
model.save_weights(path + "/model.h5")
