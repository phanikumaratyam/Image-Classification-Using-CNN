import os

import random

import warnings

warnings.filterwarnings('ignore')

import numpy as np

import pandas as pd

from scipy.misc import imread, imresize

from sklearn.preprocessing import LabelEncoder

import keras

import matplotlib.pylab as pylab

print("Libraries imported!")

seed = 27

rng = np.random.RandomState(seed)

root_dir = os.path.abspath(r'C:\Users\Phanikumar\Downloads\Age Detection - Analytics Vidhya')

print(os.path.exists(root_dir))

train = pd.read_csv(os.path.join(root_dir,'train.csv'))

test = pd.read_csv(os.path.join(root_dir,'test.csv'))

print(train.sample(10))

print(test.sample(10))

img_name = rng.choice(train.ID)

file_path = os.path.join(root_dir,'Train',img_name)

img = imread(file_path, flatten=True)

pylab.imshow(img)

pylab.axis('off')

pylab.show()

temp = []

for img_name in train.ID:
    img_path = os.path.join(root_dir,'Train',img_name)
    img = imread(img_path)
    img = imresize(img, (32,32))
    img = img.astype('float32')
    temp.append(img)
    print(len(temp))
    
train_x = np.stack(temp)

temp = [] 

for img_name in test.ID:
    img_path = os.path.join(root_dir, 'Test', img_name)
    img = imread(img_path)
    img = imresize(img, (32,32))
    img = img.astype('float32')
    temp.append(img)
    print(len(temp))
    
test_x = np.stack(temp)

train_x = train_x/255

test_x = test_x/255

#print(train.Class.value_counts(normalize=True))

label = LabelEncoder()

train_y = label.fit_transform(train.Class)

train_y = keras.utils.np_utils.to_categorical(train_y)

input_num_units = (32,32,3)

hidden_num_units = 500

output_num_units = 3

epochs = 10

batch_size = 128

pool_size = (2,2)

from keras.models import Sequential

from keras.layers import Dense, Activation, Dropout, Convolution2D, Flatten, MaxPooling2D, Reshape, InputLayer

model = Sequential([
        InputLayer(input_shape = input_num_units),
        
        Convolution2D(50,5,5, activation='relu'),
        MaxPooling2D(pool_size = pool_size),
        
        Convolution2D(100,5,5, activation = 'relu'),
        MaxPooling2D(pool_size = pool_size),
        
        Convolution2D(100,5,5, activation = 'relu'),
        
        Flatten(),
        
        Dense(output_dim = hidden_num_units, activation = 'relu'),
        Dense(output_dim = output_num_units, input_dim = hidden_num_units, activation = 'softmax')
        ])
    
model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])

trained_model_conv = model.fit(train_x,train_y,nb_epoch = epochs, batch_size = 128, validation_split = 0.2)

