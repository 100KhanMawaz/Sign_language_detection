import os
import cv2
import mediapipe as mp
import numpy as np
import pywt
import pickle
import matplotlib.pyplot as plt
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)
DATA='./Data'
char_dict = {}

def w2d(img, mode='haar', level=1):
    imArray = img
    #Datatype conversions
    #convert to grayscale
    imArray = cv2.cvtColor( imArray,cv2.COLOR_RGB2GRAY )
    #convert to float
    imArray =  np.float32(imArray)
    imArray /= 255;
    # compute coefficients
    coeffs=pywt.wavedec2(imArray, mode, level=level)

    #Process Coefficients
    coeffs_H=list(coeffs)
    coeffs_H[0] *= 0;

    # reconstruction
    imArray_H=pywt.waverec2(coeffs_H, mode);
    imArray_H *= 255;
    imArray_H =  np.uint8(imArray_H)

    return imArray_H

for dir_ in os.listdir(DATA):
    count = 0
    char_ = dir_.split('/')[-1]
    #print(char_)
    char_dict[char_] = [] #we mean that every character in a char_dict is an array

    for img_path in os.listdir(os.path.join(DATA,dir_)):
       # print(img_path)
        char_dict[char_].append(DATA + "/" + dir_ + "/" + img_path)

#print(char_dict)
count = 0
alphabet_class_numbers_dict = {}
for class_of_alphabet in char_dict.keys():
    alphabet_class_numbers_dict[class_of_alphabet]=count
    count+=1

print(alphabet_class_numbers_dict)
x = []
y = []
for class_of_alphabet,images in char_dict.items(): #there are multiple ways to iterate over a dictionary but .items() allows us to access both key and value at the same time for all indices and hence here it is givving as 2 things which are Alphabet_class which is the in our dictionary and images array which contain current_alphabet_class images the dictonary we created is kind of map<datatype,vector<datatype>> 1st element is key of 2nd element arrays
    for training_image_path in images[:1]:
        img = cv2.imread(training_image_path)
        if img is None:
            continue
        processed_img = w2d(img,'db1',5)

        # final_img = processed_img.reshape(-1,300*300*1)
        # plt.figure()
        # plt.imshow(final_img.reshape(300,300),cmap='gray')
        # plt.show()

        #print(final_img)
        #x.append(final_img)
        #y.append(alphabet_class_numbers_dict[class_of_alphabet])
        #print(processed_img)
        #print(training_image_path)

# print(len(x)) #total number of images in all class we have
# print(len(x[0])) #size of each image. which will be 300 because we fixed that during data collection only
# print(x)

f = open('data.pickle','wb')
pickle.dump({'data':x,'labels':y},f)
f.close()

import matplotlib.pyplot as plt
# for dir_ in os.listdir(DATA)[:1]:
#     for img_path in os.listdir(os.path.join(DATA,dir_))[:1]:
#         t_img = cv2.imread(os.path.join(DATA, dir_, img_path))

