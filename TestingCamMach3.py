import numpy as np 
import cv2
from PIL import Image
from keras import models
import tensorflow as tf 
from keras.models import load_model
#from keras.models import load_weights
from keras.models import model_from_json

#model = tf.keras.models.load_model("MachLearn3Out1.model")
model = tf.keras.models.load_model("MachLearn3.model")

vid = cv2.VideoCapture(0)
img_size = 224
while True:
    ret, frame = vid.read()
    img = Image.fromarray(frame,'RGB')
    img = img.resize((img_size, img_size))
    img_array = np.array(img)

    #Our keras model used a 4D tensor, (images x height x width x channel)
    #So changing dimension 128x128x3 into 1x128x128x3 
    img_array = np.expand_dims(img_array, axis=0)

    #Calling the predict method on model to predict 'me' on the image
    prediction = model.predict(img_array)
    numA = prediction[0][0]
    numB = prediction[0][1]
    print(prediction)
    font = cv2.FONT_HERSHEY_SIMPLEX
    if numA<numB:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        cv2.putText(frame,'Truck',(10,img_size-20), font, 4,(255,255,255),2,cv2.LINE_AA)
    elif numA>numB:
        cv2.putText(frame,'Car',(10,img_size-20), font, 4,(255,255,255),2,cv2.LINE_AA)
        pass
        #frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    #gray = cv2.cvtColor(frame, cv2.COLOR_RGB)
    # Display the resulting frame
    cv2.imshow('Webcam Capturing Image',frame)
    #cv2.imshow('frame',cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

vid.release()
cv2.destroyAllWindows()