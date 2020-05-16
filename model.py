import csv
import numpy as np
from scipy import ndimage

lines = []

with open('./data/driving_log.csv') as csvfile:
	reader = csv.reader(csvfile)
	for line in reader:
		lines.append(line)
		
images = []
measurements = []
flag = True
for line in lines:
	if flag==True:
		flag = False	
	else:
		source_path = line[0]
		filename = source_path.split('/')[-1]
		current_path = './data/IMG/' + filename
		image = ndimage.imread(current_path)
		images.append(image)
		measurement = float(line[3])
		measurements.append(measurement)
	
my_lines = []

with open('/home/workspace/my_data/driving_log.csv') as csvfile:
          reader = csv.reader(csvfile)
          for line in reader:
                    my_lines.append(line)
					
for line in my_lines:
          current_path = line[0]
          image = ndimage.imread(current_path)
          images.append(image)
          measurement = float(line[3])
          measurements.append(measurement)

X_train = np.array(images)
y_train = np.array(measurements)

from keras.models import Sequential 
from keras.layers import Flatten, Dense, Lambda, Cropping2D
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
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
model.fit(X_train, y_train, validation_split=0.2, shuffle=True, nb_epoch =3)

model.save('model.h5')

