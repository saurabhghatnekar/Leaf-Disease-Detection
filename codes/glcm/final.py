import cv2


import pandas as pd
from sklearn.cross_validation import  train_test_split
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from skimage.feature import greycomatrix,greycoprops


from time import sleep


abc=pd.read_excel('feature_crop.xlsx',header=None)

X=np.array((abc.as_matrix())[1:,:])
grading=["raw","good","damaged"]
Y=X[:,8]
X=X[:,0:8]
print(X[1,:])
X_train, X_test, y_train, y_test = train_test_split(X,Y,test_size=0.01,random_state=10)
y_train = y_train.astype('int')
y_test = y_test.astype('int')


K_value = 3
neigh = KNeighborsClassifier(n_neighbors=K_value, weights='uniform', algorithm='kd_tree')
neigh.fit(X_train, y_train)
sscore=neigh.score(X_test, y_test)
print('KNN accuracy=',sscore*100)


cap0 = cv2.VideoCapture(0)

cap1 = cv2.VideoCapture(1)


proList = ['contrast', 'dissimilarity', 'homogeneity', 'ASM', 'energy']

properties =np.zeros(6)
glcmMatrix = []
final=[]

x1 = 0
y1 = 0
w1 = 463
h1 = 477


x2 = 0
y2 = 145
w2 = 386
h2 = 315


while True:

    
    ret, frame0 = cap0.read()
    ret1, frame1 = cap1.read()
    print(ret,ret1)
    cv2.imshow('frame', frame0)
    cv2.imshow('frame1', frame1)
    # if i == 0:
    #     gpio.output(4,True)
    #     gpio.output(14,True)
    # sleep(3)
    print('tomato detected')
    frame0 = frame0[0:477, 0:463]
    frame1 = frame1[142:460, 0:386]


    hsv0 = cv2.cvtColor(frame0, cv2.COLOR_BGR2HSV)
    hsv1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2HSV)
    h0,s0,v0 = cv2.split(hsv0)

    h_mean0 = np.mean(h0)
    h_mean0 = np.mean(h_mean0)

    s_mean0 = np.mean(s0)
    s_mean0 = np.mean(s_mean0)

    v_mean0 = np.mean(v0)
    v_mean0 = np.mean(v_mean0)

    h1,s1,v1 = cv2.split(hsv1)
    h_mean1 = np.mean(h1)
    h_mean1 = np.mean(h_mean1)

    s_mean1 = np.mean(s1)
    s_mean1 = np.mean(s_mean1)

    v_mean1 = np.mean(v1)
    v_mean1 = np.mean(v_mean1)



    gray_image0 = cv2.cvtColor(frame0, cv2.COLOR_BGR2GRAY)
    gray_image1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)

    glcmMatrix = (greycomatrix(gray_image0, [1], [0], levels=2 ** 8))
    for j in range(0, len(proList)):
        properties[j] = (greycoprops(glcmMatrix, prop=proList[j]))

    features = np.array(
        [properties[0], properties[1], properties[2], properties[3], properties[4],h_mean0,s_mean0,v_mean0])

    glcmMatrix1 = (greycomatrix(gray_image1, [1], [0], levels=2 ** 8))

    for j in range(0, len(proList)):
        properties[j] = (greycoprops(glcmMatrix1, prop=proList[j]))

    features2 = np.array(
        [properties[0], properties[1], properties[2], properties[3], properties[4],h_mean1,s_mean1,v_mean1])

    c1 = neigh.predict([features])[0]-1
    print(grading[c1])
    c2 = neigh.predict([features2])[0]-1
    print(grading[c2])

    # sleep(2)

    k = cv2.waitKey(1) & 0xFF
    if k == ord('q'):
        break

