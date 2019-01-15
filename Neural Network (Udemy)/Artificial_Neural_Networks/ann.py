# Artificial Neural Network

# Installing Theano
# pip install --upgrade --no-deps git+git://github.com/Theano/Theano.git
# numerical computations library -it runs on CPU and GPU (GPU has bigger memory)
# go to Terminal > type "pip install theano"

# Installing Tensorflow
# pip install tensorflow
# numerical computations library - runs on CPU and GPU 
# (Theano and Tensorflow are used for neural networks research and etc. - built from scratch)
# go to Terminal > type "pip install tensorflow"

# Installing Keras
# pip install --upgrade keras

# Part 1 - Data Preprocessing
# this is a wrapper for Theano and Tensorflow - used for simple neural network codes

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


import os
os.chdir("/Users/lisa/Neural Network (Udemy)/Deep_Learning_A_Z/Volume 1 - Supervised Deep Learning/Part 1 - Artificial Neural Networks (ANN)/Section 2 - Part 1 - ANN/Artificial_Neural_Networks")
# Importing the dataset
dataset = pd.read_csv('Churn_Modelling.csv')
X = dataset.iloc[:, 3:13].values
y = dataset.iloc[:, 13].values

# Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X_1 = LabelEncoder()
X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1])
labelencoder_X_2 = LabelEncoder()
X[:, 2] = labelencoder_X_2.fit_transform(X[:, 2])
onehotencoder = OneHotEncoder(categorical_features = [1])
X = onehotencoder.fit_transform(X).toarray()
X = X[:, 1:]

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train) # scales the training data set
X_test = sc.transform(X_test) #whatever transform scale that was taken from training is applied to the test data set

# Part 2 - Now let's make the ANN!

# Importing the Keras libraries and packages
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout # will use drop out method on fist layer

# Initialising the ANN
classifier = Sequential()

# Adding the input layer and the first hidden layer (with dropout)
classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu', input_dim = 11))
classifier.add(Dropout(p=0.1)) # p is the proportion of neurons you want to disable on iterations

# Adding the second hidden layer
classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu'))

# Adding the output layer
classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))

# Compiling the ANN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Fitting the ANN to the Training set
classifier.fit(X_train, y_train, batch_size = 10, epochs = 100) #NOTE: nb_epoch is changed to epochs

# Part 3 - Making predictions and evaluating the model

# Predicting the Test set results
y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)



#Predict if the customer will leave the bank:

#Geography: France
#Credit Score: 600
#Gender: Male
#Age: 40 years old
#Tenure: 3 years
#Balance: $60000
#Number of Products: 2
#Does this customer have a credit card ? Yes
#Is this customer an Active Member: Yes
#Estimated Salary: $50000


X_test_newcust = np.array([[0,0,600,1,40,3,60000,2,1,1,50000]]) 
#IF ONE PAIR SQUARE BRACKETS (above), IT IS A COLUMN!

X_test_newcust = sc.transform(X_test_newcust)
Y_pred_newcust = classifier.predict(X_test_newcust) # Predict prob of 0.0384433


# Part 4 - Evaluating, Improving and Tuning the ANN

# Evaluating the ANN
from keras.wrappers.scikit_learn import KerasClassifier #wrapper for scikit library
from sklearn.model_selection import cross_val_score
def build_classifier(): #our own classifier function
    classifier = Sequential()
    classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu', input_dim = 11))
    classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu'))
    classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))
    classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
    return classifier #builds the classifier and spits out the classifier

classifier = KerasClassifier(build_fn = build_classifier, batch_size = 10, epochs=100) 
accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10) #n_jobs = -1 didn't work

print(accuracies)
print(accuracies.mean())
print(accuracies.std())
 #Set to build a classifier based on the build_classifier up to classifier compilation
 
 # Tuning the ANN
 # I THINK WE ARE TRYING THE DROPOUT REGULARIZATION (it's to reduce overfitting by eliminating random nodes so that it can identify indepndent correlation of variables)
# We use something called grid search
 
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV # or it could be sklearn.grid_search if it doesn;t work
def build_classifier(optimizer): #our own classifier function
    classifier = Sequential()
    classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu', input_dim = 11))
    classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu'))
    classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))
    classifier.compile(optimizer = optimizer, loss = 'binary_crossentropy', metrics = ['accuracy'])
    return classifier #builds the classifier and spits out the classifier

classifier = KerasClassifier(build_fn = build_classifier) # NOTE: we don't put batch_size (training sample size from data set) or epochs(number of cycles) because those are the parameters we are tuning!
parameters = {'batch_size':[25, 32],#(could take powers of two). String must match parameters in KerasClassifier function
              'epochs': [100,500],
              'optimizer': ['adam', 'rmsprop']} #key-value  

grid_search = GridSearchCV(estimator = classifier, 
                           param_grid = parameters,
                           scoring = 'accuracy',
                           cv = 10)

grid_search = grid_search.fit(X_train, y_train)
best_parameters = grid_search.best_params_
best_accuracy = grid_search.best_score_

###Homework: get an accuracy of over 86%

def build_classifier(optimizer,act_1, act_2, act_3): #our own classifier function
    classifier = Sequential()
    classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = act_1, input_dim = 11))
    classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = act_2))
    classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = act_3))
    classifier.compile(optimizer = optimizer, loss = 'binary_crossentropy', metrics = ['accuracy'])
    return classifier #builds the classifier and spits out the classifier

classifier = KerasClassifier(build_fn = build_classifier) # NOTE: we don't put batch_size (training sample size from data set) or epochs(number of cycles) because those are the parameters we are tuning!

# parameter batch_size could take powers of two. String must match parameters in KerasClassifier function
# key-value type

parameters = {'batch_size':[25],#KerasClassifier object parameter
              'epochs': [500],#KerasClassifier object parameter
              'optimizer': ['rmsprop'],#build_classifier object parameter
              'act_1':['relu','sigmoid'],#build_classifier object parameter
              'act_2':['relu','sigmoid'],#build_classifier object parameter
              'act_3':['relu','sigmoid']} #build_classifier object parameter  

grid_search = GridSearchCV(estimator = classifier, 
                           param_grid = parameters, #KerasClassifier object param
                           scoring = 'accuracy',
                           cv = 10)

grid_search = grid_search.fit(X_train, y_train)
best_parameters = grid_search.best_params_

# Out[117]: 
# {'act_1': 'sigmoid',
# 'act_2': 'sigmoid',
# 'act_3': 'sigmoid',
# 'batch_size': 25,
# 'epochs': 500,
# 'optimizer': 'rmsprop'}

best_accuracy = grid_search.best_score_

# Out[118]: 0.854625