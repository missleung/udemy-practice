# Convolutional Neural Network

# Installing Theano
# pip install --upgrade --no-deps git+git://github.com/Theano/Theano.git

# Installing Tensorflow
# pip install tensorflow

# Installing Keras
# pip install --upgrade keras

# Part 1 - Building the CNN

# Importing the Keras libraries and packages
from keras.models import Sequential # to initialize neural network. 
# Remember there are two ways to set neural networks: sequence of layers or graph

from keras.layers import Conv2D # first step - add convolutional layers to deal with images
from keras.layers import MaxPooling2D # to add pooling steps
from keras.layers import Flatten # flattening, where we convert pooling maps into a large vector
from keras.layers import Dense # use to add our fully connected vector into the NN

# Note: Keras uses a data structure > train/test > cat/dog > list of pictures
# Note: Because pictures are already categorized in a special structure such that keras 
# package already knows how to read in the data, we don't need to do data preprocessing,
# we can go straight to CNN

# Initialising the CNN
classifier = Sequential() # create object of a sequential 

# Step 1 - Convolution
# Reminder: Image first is coded into pixel values (black and white = 1 and 0). Convolution layer 
# is a set of feature maps. We run a set of feature detectors (a small 3x3 matrix, ie. the corner 
# of the mouth on an image of a smiley face) to the image by sliding each feature detector to all 
# 3x3 squares combinations on the image. Then take the sum element-wise multiplication of the two 
# matrices together. By the way, it's NOT the cross product so I forgot what it's called. When the
# "corner of the mouth" detector passes thru the corner of the mouth on the image, it automatically
# spits out a higher value onto the feature map.

classifier.add(Conv2D(32, (3, 3), input_shape = (64, 64, 3), activation = 'relu'))
# add the layer for convolutional layer. For the ANN, we used the Dense function which was used to
# create the full layer of the NN. 
# Note: click the cursor after the first paranthesis ( and press CTRL+I to see Help function
# We see that first argument of Conv2D is filters. filters is the number of feature detectors
# you want to use. Second argument kernel_size is the size of the matrix (in this case, 3x3) 
# of the feature detector.
# 

# Step 2 - Pooling
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# Adding a second convolutional layer
classifier.add(Conv2D(32, (3, 3), activation = 'relu')) 
# could go 32, 64, 128, etc. on feature detectors but it's gonna take long time

# Below he talked about input_shape parameter from old package... can't find the equivalent on current
# input_shape is the expected format size for each image. Note that all images
# are need to be processed to the same size since all come in different size and shapes.
# Remember that 2D is black/white image converted 1:1 on black/grey array/channel and 
# 3D means the colored image converted 1:3 on red, green, blue arrays/channels during image 
# processing part

classifier.add(MaxPooling2D(pool_size = (2, 2)))

# Step 3 - Flattening
classifier.add(Flatten())

# Step 4 - Full connection
classifier.add(Dense(units = 128, activation = 'relu'))
classifier.add(Dense(units = 1, activation = 'sigmoid'))

# Compiling the CNN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Part 2 - Fitting the CNN to the images

from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

test_datagen = ImageDataGenerator(rescale = 1./255)

training_set = train_datagen.flow_from_directory('dataset/training_set',
                                                 target_size = (64, 64),
                                                 batch_size = 32,
                                                 class_mode = 'binary')

test_set = test_datagen.flow_from_directory('dataset/test_set',
                                            target_size = (64, 64),
                                            batch_size = 32,
                                            class_mode = 'binary')

classifier.fit_generator(training_set,
                         steps_per_epoch = 8000,
                         epochs = 25,
                         validation_data = test_set,
                         validation_steps = 2000)