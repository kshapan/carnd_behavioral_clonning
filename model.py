import math
import random
import os
import csv
import numpy as np
from PIL import Image
from scipy import ndimage

samples = []

with open('/opt/training_mode/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        samples.append(line)

from sklearn.model_selection import train_test_split
train_samples, validation_samples = train_test_split(samples, test_size=0.2)

import sklearn

def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        random.shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            measurements = []
            for batch_sample in batch_samples:
                # correction for left and right camera image
                steering_angle_correction = 0.2

                # load images and measurements
                
                center_filename = batch_sample[0].split('/')[-1]
                center_path = '/opt/training_mode/IMG/' + center_filename
                
                left_filename = batch_sample[1].split('/')[-1]
                left_path = '/opt/training_mode/IMG/' + left_filename
                
                right_filename = batch_sample[2].split('/')[-1]
                right_path = '/opt/training_mode/IMG/' + right_filename
                
                image_center = np.asarray(Image.open(center_path))
                image_left = np.asarray(Image.open(left_path))
                image_right = np.asarray(Image.open(right_path))
                steering_angle = float(batch_sample[3])

                # correct angles for left and right image
                steering_angle_left = steering_angle + steering_angle_correction
                steering_angle_right = steering_angle - steering_angle_correction

                # add original and flipped images to the list of images
                images.append(image_center)
                images.append(np.fliplr(image_center))
                images.append(image_left)
                images.append(np.fliplr(image_left))
                images.append(image_right)
                images.append(np.fliplr(image_right))

                # add corresponting measurements
                measurements.append(steering_angle)
                measurements.append(-steering_angle)
                measurements.append(steering_angle_left)
                measurements.append(-steering_angle_left)
                measurements.append(steering_angle_right)
                measurements.append(-steering_angle_right)
               
            X_train = np.array(images)
            y_train = np.array(measurements)
            yield sklearn.utils.shuffle(X_train, y_train)


# Set our batch size
batch_size=32

# compile and train the model using the generator function
train_generator = generator(train_samples, batch_size=batch_size)
validation_generator = generator(validation_samples, batch_size=batch_size)            

from keras.models import Sequential 
from keras.layers import Flatten, Dense, Lambda, Cropping2D, Dropout
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D

model = Sequential()
model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160,320,3)))
model.add(Cropping2D(cropping=((50,20), (0,0))))
model.add(Conv2D(24, kernel_size=(5, 5), strides=(2,2), activation='relu'))
model.add(Conv2D(36, kernel_size=(5, 5), strides=(2,2), activation='relu'))
model.add(Conv2D(48, kernel_size=(5, 5), strides=(2,2), activation='relu'))
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
model.add(Flatten())
model.add(Dropout(0.2))
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')

model.fit_generator(
    train_generator, 
    steps_per_epoch=math.ceil(len(train_samples)/batch_size), 
    validation_data=validation_generator, 
    validation_steps=math.ceil(len(validation_samples)/batch_size), 
    epochs=10, verbose=1)

model.save('model.h5')

