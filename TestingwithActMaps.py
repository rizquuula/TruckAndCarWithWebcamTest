from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import decode_predictions
import cv2
import matplotlib.image as mpimg
from keras import backend as K
import matplotlib.pyplot as plt
import pandas as pd
#matplotlib inline
import numpy as np
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
import os
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
def cam(img_path):

    K.clear_session()
    
    model = VGG16(weights= 'imagenet')
    # print(model.summary())
    img=mpimg.imread(img_path)
    plt.imshow(img)
    
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    
    x = preprocess_input(x)
    # print(x)
    preds = model.predict(x)
    print()
    #print('preds = ',preds)
    #print('preds[0] = ',preds[0])
    predict2 = decode_predictions(preds)
    print(predict2)
    predictions = pd.DataFrame(decode_predictions(preds, 
        top=5)[0],
        columns=['col1','category','probability']).iloc[:,1:]
    print(predictions)
    print()
    argmax = np.argmax(preds[0])
    argmaxLess = np.argmax(preds[0][0])
    # print(argmax,' and ', argmaxLess)
    output = model.output[:, argmax]
    # print('Output = ',output)
    last_conv_layer = model.get_layer('block5_conv3')
    # print('last_conv_layer = ',last_conv_layer)
    grads = K.gradients(output, last_conv_layer.output)[0]
    grads1 = K.gradients(output, last_conv_layer.output)
    # print('grads = ',grads)
    # print('grads1 = ',grads1)
    pooled_grads = K.mean(grads, axis=(0, 1, 2))
    #pooled_grads2 = K.mean(grads, axis=(0, 1, 2))
    # print('pooled_grads = ',pooled_grads)
    #print('pooled_grads2 = ',pooled_grads2)
    iterate = K.function([model.input], [pooled_grads, last_conv_layer.output[0]])
    # print('model.input = ',model.input)
    # print()
    # print('last_conv_layer.output[0] = ',last_conv_layer.output[0])
    # print()
    # print('last_conv_layer.output[1] = ',last_conv_layer.output[1])
    # print()
    # print('last_conv_layer.output = ',last_conv_layer.output)
    # print()
    # print('iterate = ',iterate)
    #print('iterate([x]) = ',iterate([x]))
    pooled_grads_value, conv_layer_output_value = iterate([x])
    for i in range(512):
        conv_layer_output_value[:, :, i] *= pooled_grads_value[i]
        #print(pooled_grads_value[i])        
        #print()
    #print(conv_layer_output_value)
    heatmap = np.mean(conv_layer_output_value, axis=-1)
    #print('heatmap1 = ',heatmap)
    cv2.imshow('Heatmap1',heatmap)
    heatmap = np.maximum(heatmap, 0)
    #print('heatmapmax = ',heatmap)
    heatmap /= np.max(heatmap)
    #print('heatmap /= = ',heatmap)
    
    img = cv2.imread(img_path)
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    cv2.imshow('Heatmap2',heatmap)
    heatmap = np.uint8(255 * heatmap)
    cv2.imshow('Heatmap3',heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    cv2.imshow('Heatmap4',heatmap)
    hif = .8
    superimposed_img = heatmap * hif + img
    cv2.imshow('SuperimposedImg',superimposed_img)
    # print('superimposed_img = ', superimposed_img)
    output = 'output.jpeg'
    cv2.imwrite(output, superimposed_img)
    img=cv2.imread(output)
    cv2.imshow('Result',img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    #plt.axis('off')
    #plt.title(predictions.loc[0,'category'].upper())
    return None
def img(view):
    img = cv2.imread(view)
    cv2.imshow('image',img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    

#img('TestingMach3/mobil-kebakar.jpg')
#cam('TestingMach3/mobil-kebakar.jpg')

#img('/home/linkgish/Desktop/MachLearn3/TestingMach3/145195.jpg')
#cam('/home/linkgish/Desktop/MachLearn8_VGG16example/143302.jpg')
cam('/home/linkgish/Desktop/MachLearn3/ValidationData/truck/206967646.jpg')
