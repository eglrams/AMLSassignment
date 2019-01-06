# AMLSassignment

There are several libraries which need to be installed to run the files in this repository:
numpy
opencv
tensorflow
tflearn

Start with running 'Task1.py'. This performs the noisy image removal.

The training and testing images should be moved manually into directories 'trainImages' and 'testImages'. 80% is training data and 
20% is testing data. I used the ;ast 20% of the images as test data.

The files 'Task2.py', 'Task2b.py', 'Task2c.py', 'Task2d.py' are the binary classification tasks in order: smile/no smile, young/old,
glasses/no glasses, human/cartoon. These can be run in any order. These files preprocess the data, makes a training and test array,
build the CNN, train on the training data, predicts the test data and writes the predictions to a csv file titled the same as the 
file, ie. 'Task2.csv', 'Task2b.csv'. The validation data split is set in the "CNN data input" stage. The file 'AccuracyPrediction.py' 
prints out a confusion matrix and the accuracy of the test predictions. The filename to be opened for the predictions must be altered 
in the file manually, it is set to open 'Task2.csv' for prediction data currently. The attribute to compare in 'attribute_list.csv' 
must be altered to match with the right prediction file. The variable 'attribute_label' must be changed to do that.

attribute_label = 2 "Glasses/No glasses"
attribute_label = 3 "Smile/No smile"
attribute_label = 4 "Young/Old"
attribute_label = 5 "Human/Cartoon"

The multiclass classification task is ran in 'Task2e.py'. This preprocesses the data, makes a training and test array,
builds the CNN, trains on the training data, predicts the test data and writes the predictions to a csv file titled 'Task2e.csv'. 
The file 'AccuracyPredictionMultiClass.py' prints out a confusion matrix and the accuracy of the test predictions for a 
multiclass problem with 6 classes.

The classification task files can be ran in any order, but the accuracy prediction file can be ran only after the specific classification 
file has ran.