import cv2
import imutils
import numpy as np
from os.path import dirname, join

prototxt_path = join(dirname(__file__), "deploy.prototxt")

model_path = join(dirname(__file__), "res10_300x300_ssd_iter_140000_fp16.caffemodel")
model = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)

image = cv2.imread("BillGates.jpg") #Read the image
image = imutils.resize(image, width=480) #Set the width as 480
cv2.imshow("Original",image) #Show the image
cv2.waitKey(0)

h, w = image.shape[:2] #We have height and width

kernel_width = (w // 7) | 1 #Gaussian blur kernel size depends on width of original image
kernel_height = (h // 7) | 1 #Gaussian blur kernel size depends on height of original image

blob = cv2.dnn.blobFromImage(image, 1.0, (300, 300), (104.0, 177.0, 123.0)) #Creates 4-dimensional blob from image (https://docs.opencv.org/master/d6/d0f/group__dnn.html#ga29f34df9376379a603acd8df581ac8d7)

model.setInput(blob) #This line help us to detect face

output = np.squeeze(model.forward()) #Perform inference and get the result

for i in range(0, output.shape[0]):
    probability = output[i, 2]


    if probability > 0.5: #No need to check weak probabilities

        box = output[i, 3:7] * np.array([w, h, w, h]) #Calculate the coordinates of the box
        start_x, start_y, end_x, end_y = box.astype(int) #Convert to int
        face = image[start_y: end_y, start_x: end_x] #Get the face
        face = cv2.GaussianBlur(face, (kernel_width, kernel_height), 0) #Apply Gaussian blur
        image[start_y: end_y, start_x: end_x] = face #Put the blurred face into the original image
        cv2.imwrite('blurred.jpg', image) #Create the blurred image

frame=cv2.imread("blurred.jpg") #Read the blurred image
cv2.imshow("Blurred",frame) #Show the blurred image
cv2.waitKey(0)
