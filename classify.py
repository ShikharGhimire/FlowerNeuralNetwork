#Importing the necessary packages
from keras.preprocessing.image import img_to_array
from keras.models import load_model
import numpy as np
import argparse
import imutils
import cv2
import pickle
import os

# construct the argument parse and parse the arguments

# construct the argument parse to use the files in command line 
#To configure command line arguments to run in the python script from spyder go to:
#run->configuration per file->command line options and enter different arguments (in this case -dataset -model -labelbin -plot)
ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model",required=True,
	help="path to trained model model")
ap.add_argument("-l", "--labelbin",required=True,
	help="path to label binarizer")
ap.add_argument("-i", "--image",required=True,
	help="path to input image")
args = vars(ap.parse_args())

#Loading the images
image = cv2.imread(args["image"])
output = image.copy() #For displaying purposes

#Image preprocessing for classification
image = cv2.resize(image, (96, 96))
image = image.astype("float") / 255.0
image = img_to_array(image)
image = np.expand_dims(image, axis=0)

#Loading our neural network with saved model and labelbinariser
print("Loading our own Convolutional Neural Network")
model = load_model(args["model"])
lb = pickle.loads(open(args["labelbin"], "rb").read())

#In this step we classify the input image
print("Classifying the images")
prediction = model.predict(image)[0]
idx = np.argmax(prediction)
label = lb.classes_[idx]

filename = args["image"][args["image"].rfind(os.path.sep) + 1:]
correct = "correct" if filename.rfind(label) != -1 else "incorrect"
 
# build the label and draw the label on the image
label = "{}: {:.2f}%".format(label, prediction[idx] * 100,correct)
output = imutils.resize(output, width=400)
cv2.putText(output, label, (10, 25),  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2) #For text display
 
# show the output image
print("[INFO] {}".format(label))
cv2.imshow("Output", output)
cv2.waitKey(0)

#python classify.py --model odel --labelbin abelbin --image examples/marigold.jpg Line for the command terminal