import cv2
import tensorflow
import numpy as np
import os
import csv


train_dir = 'trainImages'
test_dir = 'testImages'
img_size = 64
lr = 1e-3

model_name = 'smilevsnosmile-{}-{}.model'.format(lr, '2convolutionsCNN')

def label_img(img, attribute_label):
    # extract label from csv file
    with open('attribute_list.csv', mode='r') as attribute_list: # open attribute csv in read mode
        label_file = csv.reader(attribute_list, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        for row in label_file: # search every row
            if img.split('.')[-2] == row[0]: # if filename is found
                if row[attribute_label] == 1: # Only works for binary tasks
                    return [1,0] # One-hot array where [1,0] = smile
                else:
                    return [0,1] # [0,1] = no_smile
    label_file.close()

def create_train_data(attribute_label):
    # create array for training data according to classification task
    training_data = []
    print('Creating training data...')
    for img in os.listdir(train_dir): # Resize each image and label them with appropriate class_label
        label = label_img(img, attribute_label) 
        path = os.path.join(train_dir, img)
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE) # read images and convert to greyscale
        img = cv2.resize(img, (img_size,img_size)) # resize to smaller image size
        training_data.append([np.array(img), np.array(label)])
    np.save('traindata.npy', training_data) # save data in numpy array
    print('Training data created.')
    return training_data

def process_test_data():
    #create array for test set predictions
    testing_data = []
    for img in os.listdir(test_dir):
        path = os.path.join(train_dir, img)
        img.num = img.split('.')[0]
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        testing_data.append([np.array(img), img_num])

    np.save('test_data.npy', testing_data)
    return testing_data

train_data = create_train_data(3) # attribute label = 3 (smile column)


# Build CNN from tensorflow and tflearn libraries
# 2 layer  CNN with fully connected layer and output layer

import tflearn
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression

print('Building model...')

import tensorflow as tf
tf.reset_default_graph() # resets graph if new model built

cnn = input_data(shape=[None, img_size, img_size, 1], name='input')

cnn = conv_2d(cnn, 32, 2, activation='relu')
cnn = max_pool_2d(cnn, 2)

cnn = conv_2d(cnn, 64, 2, activation='relu')
cnn = max_pool_2d(cnn, 2)

cnn = fully_connected(cnn, 1024, activation='relu') #fully connected layer
cnn = dropout(cnn, 0.8)

cnn = fully_connected(cnn, 2, activation='softmax') # output layer
cnn = regression(cnn, optimizer='adam', learning_rate=lr, loss='categorical_crossentropy', name='targets')

model = tflearn.DNN(cnn, tensorboard_dir='log') # define model

print('Model built.')

if os.path.exists('{}.meta'.format(model_name)): # load CNN if already exists
    model.load(model_name)
    print('Existing model loaded.')

# input data into CNN

print('Inputting data into model...')

# Split into training and validation sets
train = train_data[:-622] 
test = train_data[-622:]

X = np.array([i[0] for i in train]).reshape(-1,img_size,img_size,1) # Reshape to smaller image size
Y = np.array([i[1] for i in train]).reshape(-1,2)

test_x = np.array([i[0] for i in test]).reshape(-1,img_size,img_size,1)
test_y = np.array([i[1] for i in test]).reshape(-1,2)

print('Fitting model...')

# Model fit with 5 epochs, a training set of 2488 images, a validation set of 622 images 
model.fit({'input': X}, {'targets': Y}, n_epoch=5, validation_set=({'input': test_x}, {'targets': test_y}), 
    snapshot_step=500, show_metric=True, run_id=model_name)








