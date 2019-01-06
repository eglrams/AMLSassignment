# FIlename: AccuracyPredictionMultiClass.py
# Author: Etienne Ramsay
# Description: Calculates the inference accuracy for the multiclass classification task
# and appends it to the respective csv file. A confusion matrix is also made.

import csv
import numpy as np
import os
import tensorflow as tf

test_dir = 'testImages'
attribute_label = 1

# Build confusion matrix from list of labels and predictions
img_num = []
predictions = []
with open('Task2e.csv', mode='r') as Task2: # Read prediction from csv file
    TaskReader = csv.reader(Task2, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    next(TaskReader) # Skip first line with inference accuracy
    for row in TaskReader:
        if not(row): # check if row is empty
            continue
        else:
            img_num.append(row[0])
            predictions.append(int(row[1])) # add prediction to array
Task2.close()
np.array(predictions)

print('Image number array:')
print(img_num)
print('Prediction array:')
print(predictions)
    
labels = []  # Create labels array for confusion matrix
with open('attribute_list.csv', mode='r') as attribute_list, open('Task2.csv', mode ='r'): # open attribute csv in read mode
    attribute_file = csv.reader(attribute_list, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    for row in attribute_file: # search every row
        for i in img_num:
            if i == row[0]:
                if row[attribute_label] == '-1':
                    num = 4
                else:
                    num = int(row[attribute_label])
                labels.append(num)
attribute_list.close()

np.array(labels)

print('Label array:')
print(labels)

confusion_matrix = tf.confusion_matrix(labels, predictions)
with tf.Session():
    print(tf.Tensor.eval(confusion_matrix))
    correct1 = (tf.Tensor.eval(confusion_matrix[0][0]))
    correct2 = (tf.Tensor.eval(confusion_matrix[1][1]))
    correct3 = (tf.Tensor.eval(confusion_matrix[2][2]))
    correct4 = (tf.Tensor.eval(confusion_matrix[3][3]))
    correct5 = (tf.Tensor.eval(confusion_matrix[4][4]))
    correct6 = (tf.Tensor.eval(confusion_matrix[5][5]))

inference_accuracy = (correct1 + correct2 + correct3 + correct4 +correct5 +correct6)/777
print(inference_accuracy)

with open('Task2e.csv', mode='a') as Task2: # Write inference accuracy to file
    TaskWriter = csv.writer(Task2, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    TaskWriter.writerow(str(inference_accuracy))
Task2.close()
