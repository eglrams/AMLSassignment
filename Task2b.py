# Filename: Task2b.py
#Author: Etienne Ramsay
# Description: Building CNN for old vs young binary classification and running prediction.

import cv2
import tensorflow
import numpy as np
import os
import csv
import Task2
from Task2 import create_train_data
from Task2 import create_test_data

train_dir = 'trainImages'
test_dir = 'testImages'
img_size = 64
lr = 1e-3

model_name = 'YoungVsOld-{}-{}.model'.format(lr, 'ModelCNN')

if __name__ == '__main__':
    if os.path.exists('traindata4.npy'):
        train_data = np.load('traindata4.npy')
    else:
        train_data = create_train_data(4) # attribute label = 4 (old/young column)

    # Build CNN from tensorflow and tflearn libraries
    # CNN with 6 convolution layers and one fully connected layer

    import tflearn
    from tflearn.layers.conv import conv_2d, max_pool_2d
    from tflearn.layers.core import input_data, dropout, fully_connected
    from tflearn.layers.estimator import regression

    print('Building model...')

    import tensorflow as tf
    tf.reset_default_graph() # resets graph if new model built

    cnn = input_data(shape=[None, img_size, img_size, 1], name='input')

    cnn = conv_2d(cnn, 32, 2, activation='tanh')
    cnn = max_pool_2d(cnn, 2)

    cnn = conv_2d(cnn, 64, 2, activation='tanh')
    cnn = max_pool_2d(cnn, 2)

    cnn = conv_2d(cnn, 32, 2, activation='tanh')
    cnn = max_pool_2d(cnn, 2)

    cnn = conv_2d(cnn, 64, 2, activation='tanh')
    cnn = max_pool_2d(cnn, 2)

    cnn = conv_2d(cnn, 32, 2, activation='tanh')
    cnn = max_pool_2d(cnn, 2)

    cnn = conv_2d(cnn, 64, 2, activation='tanh')
    cnn = max_pool_2d(cnn, 2)
    
    cnn = fully_connected(cnn, 1024, activation='tanh') #fully connected layer
    cnn = dropout(cnn, 0.6)

    cnn = fully_connected(cnn, 2, activation='softmax') # output layer
    cnn = regression(cnn, optimizer='adam', learning_rate=lr, loss='categorical_crossentropy', name='targets') # Classes are mutually exclusive, loss selection is appropriate

    model = tflearn.DNN(cnn, tensorboard_dir='log') # define model

    print('Model built.')

    model.save(model_name)

    #############################################################################################################


    #if os.path.exists('{}.meta'.format(model_name)): # load CNN if already exists
        #model.load(model_name)
        #print('Existing model loaded.')

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

    # Model fit with 3 epochs, a training set of 2488 images, a validation set of 622 images 
    model.fit({'input': X}, {'targets': Y}, n_epoch=3, validation_set=({'input': test_x}, {'targets': test_y}), 
        snapshot_step=500, show_metric=True, run_id=model_name)


    # Predictions for test data
    print('Running prediction...')

    if os.path.exists('testdata.npy'): # check if test data array exists
        test_data = np.load('testdata.npy')
    else: 
        test_data = create_test_data()


    with open('Task2b.csv', mode='w') as Task2b: # Writing to csv file 
        TaskWriter = csv.writer(Task2b, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        placeholder = ['Inference Accuracy']
        csvData = []
        TaskWriter.writerow(placeholder) # Placeholder for inference accuracy
        for data in test_data: # Run prediction for each image and store results
            img_num = data[1]
            img_data = data[0]
            print('Image name: ', img_num)
            img_data_shaped = img_data.reshape(1,img_size,img_size,1)
            model_out = model.predict(img_data_shaped)
            print('Output: ', model_out)
            print('Output of first column: ', model_out[0][0])
            if np.argmax(model_out) == 0: # [1,0] = smile, argmax should be for first element
                img_pred = 1
                print('Young person Predicted')
                print('Argmax is ', np.argmax(model_out))
            else:
                img_pred = 0
                print('Old person predicted')
                print('Argmax is ', np.argmax(model_out))
            csvData.append([img_num,img_pred])
            #TaskWriter.writerow('{},{}'.format(img_num,model_out[1])) # Write filename and prediction to row
        TaskWriter.writerows(csvData)
    Task2b.close()

    print('Printing csvData length')
    print(len(csvData))

    print('Prediction completed')
