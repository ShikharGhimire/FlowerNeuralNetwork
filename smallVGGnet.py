#Importing dataset for data modification
import numpy as np
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import tensorflow as tf
from keras.layers.core import Activation

#Importing the deep learning libraries and packages
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Dropout #Not used

#We are going to use smallVGGnet neural network architecture 
class NeuralNetwork:
    def build(height,width,depth,classes):
        model = Sequential() #Initlialising the neural network
        input_shape = (height,width,depth) #Channels RGB
        #First convolution and pooling
        model.add(Convolution2D(32,(3,3),input_shape=input_shape,padding='same',activation = 'relu'))
        model.add(MaxPooling2D(pool_size=(2,2)))
        model.add(Dropout(0.25))
        #Second convolution and pooling
        model.add(Convolution2D(64,(3,3),padding='same',activation='relu')) #Changing the filter from 34 to 64 as we go deeper into the network
        model.add(MaxPooling2D(pool_size=(2,2)))
        model.add(Dropout(0.25))
        
        #Third layer of convoluting and pooling
        model.add(Convolution2D(64,(3,3),padding='same',activation='relu')) #Increasing the filters
        model.add(MaxPooling2D(pool_size=(2,2)))
        model.add(Dropout(0.50)) #More details are learned so therefore dropout function is increased to avoid overfitting
        
        #Flattening
        model.add(Flatten())
        #Connecting the layers
        model.add(Dense(1000)) #Connecting each layers with 1000 neurons
        
        #Classifier
        model.add(Dense(classes)) #Depending on the classes there will be certain number of nodes 
        model.add(Activation("softmax")) #More than 2 category of output
        
        return model #Returns our neural network
    
        
        
        
        