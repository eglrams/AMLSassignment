# Filename: Task2e.py
#Author: Etienne Ramsay
# Description: Building CNN for hair colour multiclass classification and running prediction.

import cv2
import tensorflow
import numpy as np
import os
import csv
import Task2


train_dir = 'trainImages'
test_dir = 'testImages'
img_size = 64
lr = 1e-3

model_name = 'HairColour-{}-{}.model'.format(lr, 'ModelCNN')

# New functions suited for a multiclass classification task

def label_img(img, attribute_label):
    # extract label from csv file
    with open('attribute_list.csv', mode='r') as attribute_list: # open attribute csv in read mode
        label_file = csv.reader(attribute_list, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        for row in label_file: # search every row
            if img.split('.')[-2] == row[0]: # if filename is found
                if row[attribute_label] == '0': # Bold 
                    print('Output of attribute row: ', row[attribute_label])
                    return [1,0,0,0,0,0]
                elif row[attribute_label] == '1': # Blond hair
                    print('Output of attribute row: ', row[attribute_label])
                    return [0,1,0,0,0,0] 
                elif row[attribute_label] == '2': # Ginger hair
                    print('Output of attribute row: ', row[attribute_label])
                    return [0,0,1,0,0,0]
                elif row[attribute_label] == '3': # Brown hair
                    print('Output of attribute row: ', row[attribute_label])
                    return [0,0,0,1,0,0]
                elif row[attribute_label] == '4': # Black hair
                    print('Output of attribute row: ', row[attribute_label])
                    return [0,0,0,0,1,0]
                elif row[attribute_label] == '5': # Grey hair
                    print('Output of attribute row: ', row[attribute_label])
                    return [0,0,0,0,0,1]
                else: # For mislabeled images with '-1' label
                    return [0,0,0,0,1,0] # Return black hair. most common hair colour
    attribute_list.close()

def create_train_data(attribute_label):
    # create array for training data according to classification task
    training_data = []
    traindata_filename = 'traindata{}.npy'.format(attribute_label)
    print('Creating training data...')
    for img in os.listdir(train_dir): # Resize each image and label them with appropriate class_label
        label = label_img(img, attribute_label) 
        path = os.path.join(train_dir, img)
        img = cv2.imread(path) # read images 
        img = cv2.resize(img, (img_size,img_size)) # resize to smaller image size
        training_data.append([np.array(img), np.array(label)])
        print('Label: ', label)
    np.save(traindata_filename, training_data) # save data in numpy array
    print('Training data created.')
    return training_data

def create_test_data():
    #create array for test set predictions
    testing_data = []
    for img in os.listdir(test_dir):
        path = os.path.join(test_dir, img)
        img_num = img.split('.')[0]
        img = cv2.imread(path) # read images 
        img = cv2.resize(img, (img_size,img_size)) # resize to smaller image size
        testing_data.append([np.array(img), np.array(img_num)])

    np.save('testdata_multi.npy', testing_data)
    return testing_data

######################################################################

if __name__ == '__main__':
    if os.path.exists('traindata1.npy'):
        train_data = np.load('traindata1.npy')
    else:
        train_data = create_train_data(1) # attribute label = 1 (Hair cloumn)

    # Build CNN from tensorflow and tflearn libraries
    # CNN based on "ImageNet Classification with Deep Convolutional Neural Networks"
    # As described in Section 4 of the paper
    # 48-128-192-192-128-2048-2048

    import tflearn
    from tflearn.layers.conv import conv_2d, max_pool_2d
    from tflearn.layers.core import input_data, dropout, fully_connected
    from tflearn.layers.estimator import regression

    print('Building model...')

    import tensorflow as tf
    tf.reset_default_graph() # resets graph if new model built

    cnn = input_data(shape=[None, img_size, img_size, 3], name='input')

    cnn = conv_2d(cnn, 48, 6, activation='tanh')
    cnn = max_pool_2d(cnn, 6)

    cnn = conv_2d(cnn, 128, 6, activation='tanh')
    cnn = max_pool_2d(cnn, 6)

    cnn = conv_2d(cnn, 192, 6, activation='tanh')
    cnn = max_pool_2d(cnn, 6)

    cnn = conv_2d(cnn, 192, 6, activation='tanh')
    cnn = max_pool_2d(cnn, 6)

    cnn = conv_2d(cnn, 128, 6, activation='tanh')
    cnn = max_pool_2d(cnn, 6)
    
    cnn = fully_connected(cnn, 2048, activation='tanh') #fully connected layer
    cnn = dropout(cnn, 0.6)

    cnn = fully_connected(cnn, 2048, activation='tanh') #fully connected layer
    cnn = dropout(cnn, 0.6)

    cnn = fully_connected(cnn, 6, activation='softmax') # output layer
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

    X = np.array([i[0] for i in train]).reshape(-1,img_size,img_size,3) # Reshape to smaller image size
    Y = np.array([i[1] for i in train]).reshape(-1,6)

    test_x = np.array([i[0] for i in test]).reshape(-1,img_size,img_size,3)
    test_y = np.array([i[1] for i in test]).reshape(-1,6)

    print('Fitting model...')

    # Model fit with 3 epochs, a training set of 2488 images, a validation set of 622 images 
    model.fit({'input': X}, {'targets': Y}, n_epoch=3, validation_set=({'input': test_x}, {'targets': test_y}), 
        snapshot_step=500, show_metric=True, run_id=model_name)


    # Predictions for test data
    print('Running prediction...')

    if os.path.exists('testdata_multi.npy'): # check if test data array exists
        test_data = np.load('testdata_multi.npy')
    else: 
        test_data = create_test_data()


    with open('Task2e.csv', mode='w') as Task2e: # Writing to csv file 
        TaskWriter = csv.writer(Task2e, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        placeholder = ['Inference Accuracy']
        csvData = []
        TaskWriter.writerow(placeholder) # Placeholder for inference accuracy
        for data in test_data: # Run prediction for each image and store results
            img_num = data[1]
            img_data = data[0]
            print('Image name: ', img_num)
            img_data_shaped = img_data.reshape(1,img_size,img_size,3)
            model_out = model.predict(img_data_shaped)
            print('Output: ', model_out)
            if np.argmax(model_out) == 0: # 6 length array for hair type
                img_pred = 0
                print('Bold person Predicted')
                print('Argmax is ', np.argmax(model_out))
            elif np.argmax(model_out) == 1:
                img_pred = 1
                print('Blond Hair predicted')
                print('Argmax is ', np.argmax(model_out))
            elif np.argmax(model_out) == 2:
                img_pred = 2
                print('Ginger Hair predicted')
                print('Argmax is ', np.argmax(model_out))
            elif np.argmax(model_out) == 3:
                img_pred = 3
                print('Brown Hair predicted')
                print('Argmax is ', np.argmax(model_out))
            elif np.argmax(model_out) == 4:
                img_pred = 4
                print('Black Hair predicted')
                print('Argmax is ', np.argmax(model_out))
            elif np.argmax(model_out) == 5:
                img_pred = 5
                print('Grey Hair predicted')
                print('Argmax is ', np.argmax(model_out))
            csvData.append([img_num,img_pred])
        TaskWriter.writerows(csvData)
    Task2e.close()

    print('Printing csvData length')
    print(len(csvData))

    print('Prediction completed')
