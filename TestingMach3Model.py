import os
import cv2
import tensorflow as tf 

whichModel = ['Truck','Mobil']
directory = 'TestingMach3/'    
isi = os.listdir(directory)

def testingIMG(filepath):
    desired_size = 224
    im = cv2.imread(filepath)#, cv2.IMREAD_GRAYSCALE)
    #new_array = cv2.resize(img_array, (desired_size,desired_size))
    old_size = im.shape[:2] # old_size is in (height, width) format

    ratio = float(desired_size)/max(old_size)
    new_size = tuple([int(x*ratio) for x in old_size])

    # new_size should be in (width, height) format

    im = cv2.resize(im, (new_size[1], new_size[0]))

    delta_w = desired_size - new_size[1]
    delta_h = desired_size - new_size[0]
    top, bottom = delta_h//2, delta_h-(delta_h//2)
    left, right = delta_w//2, delta_w-(delta_w//2)

    #color = [255, 255, 255]
    color = [0, 0, 0]
    new_im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT,
        value=color)
    return new_im.reshape(-1,desired_size,desired_size,3)

model = tf.keras.models.load_model("MachLearn3Out1.model")

model.summary()
# 22 22 64
print(22*22*64)
#prediction = model.predict([testingIMG(test_data_dir)])
#print(prediction)

for i in isi:
    
    data = str(directory+i)
    prediction = model.predict([testingIMG(data)])
    
    print(data,whichModel[int(prediction[0][1])]) 
        #' = ',gender[int(prediction[0][0])])