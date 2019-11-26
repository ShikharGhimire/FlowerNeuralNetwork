#Setting up the libraries
import matplotlib
matplotlib.use("Agg") #Using agg backend to put photo 

#Importing all the libraries and packages
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from imagesearch.smallVGGnet import NeuralNetwork #Importing our neural network class
from sklearn.preprocessing import LabelBinarizer
from tqdm import tqdm
import numpy as np
import argparse
import cv2
import tensorflow as tf
import os
import pickle

# construct the argument parse to use the files in command line 
#To configure command line arguments to run in the python script from spyder go to:
#run->configuration per file->command line options and enter different arguments (in this case -dataset -model -labelbin -plot)

ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset",required=True,
	help="path to input dataset")
ap.add_argument("-m", "--model",required=True,
	help="path to output model")
ap.add_argument("-l", "--labelbin",required=True,
	help="path to output label binarizer")
args = vars(ap.parse_args())

#For memory growth 
config = tf.ConfigProto()
config.gpu_options.allow_growth = True #GPU memory uses
session = tf.Session(config=config) 

data= [] #List for the flower data
labels = [] #List for the labels(name of flowers)

image_size = 96
#Flowers directory
daisy_flower = (r'C:\Users\shikh\Desktop\flowers-recognition\flowers\daisy')
marigold_flower = (r'C:\Users\shikh\Desktop\flowers-recognition\flowers\marigold')
rose_flower = (r'C:\Users\shikh\Desktop\flowers-recognition\flowers\rose')
sunflower_flower = (r'C:\Users\shikh\Desktop\flowers-recognition\flowers\sunflower')
tulip_flower = (r'C:\Users\shikh\Desktop\flowers-recognition\flowers\tulip')
dandelion_flower = (r'C:\Users\shikh\Desktop\flowers-recognition\flowers\dandelion')
night_jasmine = (r'C:\Users\shikh\Desktop\flowers-recognition\flowers\nightjasmine')
rhododendron = (r'C:\Users\shikh\Desktop\flowers-recognition\flowers\rhododendron')
background = (r'C:\Users\shikh\Desktop\flowers-recognition\flowers\background')

def flower_labels(img,flower_name):
    return flower_name

def train_data(flower_name,directory):
    for img in tqdm(os.listdir(directory)):
        label = flower_labels(img,flower_name)
        path = os.path.join(directory,img)
        img = cv2.imread(path,cv2.IMREAD_COLOR) #Getting numpy arrays of pixel value of images
        img = cv2.resize(img,(image_size,image_size))#resetting the image into 96,96
        
        #Assigning it into lists
        data.append(np.array(img))
        labels.append(str(label))
        
        
#Extracting colour values and finding out the number of datas in the array       
train_data('Daisy-डेजी',daisy_flower)
print(len(data))

train_data('Marigold-सयपत्री',marigold_flower)
print(len(data))

train_data('Rose-गुलाफ',rose_flower)
print(len(data))

train_data('Sunflower-सूर्यमुखी',sunflower_flower)
print(len(data))

train_data('Tulip-घण्टी फुल',tulip_flower)
print(len(data))

train_data('Dandelion-डेन्डेलिओन',dandelion_flower)
print(len(data))

train_data('nightjasmine-पारिजात',night_jasmine)
print(len(data))

train_data('Rhododendron-लालिगुरास',rhododendron)
print(len(data))

train_data('Background',background)
print(len(background))

init_g = tf.global_variables_initializer()
init_l = tf.local_variables_initializer()
with tf.Session() as sess:
    sess.run(init_g)
    sess.run(init_l)

data = np.array(data)/255#Dividing it by 255 to get 0 and 1 value in the data matrix
labels = np.array(labels)
labelBinarised = LabelBinarizer()
labels = labelBinarised.fit_transform(labels) #Binarising the labels

#Splitting the training data into training and testing set
X_train,X_test,Y_train,Y_test = train_test_split(data,labels,test_size=0.25,random_state=0)

#Augmenting the X training data so that there won't be any overfitting and generating new sets of images
datagen = ImageDataGenerator(featurewise_center = False,
                             samplewise_center=False,
                             featurewise_std_normalization=False,
                             samplewise_std_normalization=False,
                             zca_whitening=False,
                             zoom_range=0.1,
                             width_shift_range=0.2,#Shifting image vertically
                             height_shift_range=0.2,#Shifting image vertically
                             horizontal_flip=True,#randomly flipping the image
                             vertical_flip=False) #randomly flipping the image

print("Training the model")

model = NeuralNetwork.build(height=96,width=96,depth=3,classes = len(labelBinarised.classes_))
#Compiling the model
model.compile(loss="categorical_crossentropy", optimizer='Adam',metrics=["accuracy"])

batch_size=15 
epochs = 500 #Training for 500 epochs
#Making prediciton using the test set
predicition = model.fit_generator(datagen.flow(X_train, Y_train, batch_size = batch_size),
                                  validation_data = (X_test,Y_test), steps_per_epoch=len(X_train) // batch_size,epochs =epochs,verbose=1)

#Saving this model to the disk
print("Saving the network")
model.save(args["model"])

#Saving the binarised label to the disk
print("Info for seralised label binariser")
saved = open(args["labelbin"], "wb")
saved.write(pickle.dumps(labelBinarised))
saved.close()