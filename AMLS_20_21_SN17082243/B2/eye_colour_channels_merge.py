#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from skimage.io import imread, imshow


# In[ ]:


image = imread('./datasets/cartoon_set/img/0.png')
imshow(image)


# In[ ]:


image.shape


# # Data Separation

# We define directions to acces the images

# In[ ]:


global basedir, image_paths, target_size
basedir = './datasets/cartoon_set'
images_dir = os.path.join(basedir,'img')
labels_filename = 'labels.csv'


# The extract_feature_label() is a function used to extract images features and store them in an array. And also store the labels in another array.

# In[ ]:


def extract_features_labels():
    
    image_paths = [os.path.join(images_dir, l) for l in os.listdir(images_dir)] # Créer un path en incluant le nom de chaque mage ?
    target_size = None
    labels_file = open(os.path.join(basedir, labels_filename), 'r')
    lines = labels_file.readlines()
    eye_labels = {line.split(',')[0] : int(line.split(',')[1]) for line in lines[1:]}
    if os.path.isdir(images_dir): # SI le Path existe 
        all_features = []
        all_labels = []
        for img_path in image_paths: # Pour chaque image 
            file_name= img_path.split('.')[1].split('/')[-1]

            # load image
            img = imread(img_path)
            
            feature_matrix = np.zeros((500,500))
            
            for i in range(0,img.shape[0]):
                for j in range(0,img.shape[1]):
                    feature_matrix[i][j] = ((int(image[i,j,0]) + int(image[i,j,1]) + int(image[i,j,2]))/3)
            
            all_features.append(features)
            all_labels.append(eye_labels[file_name])

    landmark_features = np.array(all_features)
    eye_labels = np.array(all_labels)  
    return landmark_features, smile_labels


# We have a 3D matrix of dimension (500 x 500 x 4) where 500 is the height, 500 is the width and 4 is the number of channels. To get the average pixel values, we will use a for loop:

# The new matrix will have the same height and width but only 1 channel. Now we can follow the same steps that we did in the previous section. We append the pixel values one after the other to get a 1D array:

# In[ ]:


def get_data():

    X, y = extract_features_labels()
    Y = np.array([y, -(y - 1)]).T
    tr_X = X[:7000]
    tr_Y = Y[:7000]
    te_X = X[7000:]
    te_Y = Y[7000:]

    return tr_X, tr_Y, te_X, te_Y


# In[ ]:


tr_X, tr_Y, te_X, te_Y= get_data()


# In[ ]:


tr_X.shape, te_X.shape


# # Model 
# 
# 

# ## Linear SVC

# In[ ]:


from sklearn.metrics import classification_report,accuracy_score
from sklearn import svm


# In[ ]:


def img_SVM(training_images, training_labels, test_images, test_labels):
    classifier = svm.SVC(kernel = 'linear')
    classifier.fit(training_images, training_labels)
    pred = classifier.predict(test_images)
    print("Accuracy:", accuracy_score(test_labels, pred))

    print(pred)
    return pred

pred=img_SVM(tr_X.reshape((7000, 68*2)), list(zip(*tr_Y))[0], te_X.reshape((815, 68*2)), list(zip(*te_Y))[0])
#pred=img_SVM(tr_X, list(zip(*tr_Y))[0], te_X, list(zip(*te_Y))[0])


# # K-Neigbours Classifier

# In[ ]:


from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
from sklearn.model_selection import train_test_split


# In[ ]:


X, y = extract_features_labels()


# In[ ]:


X.shape, y.shape


# In[ ]:


X = X.reshape(7815,68*2)
X.shape


# In[ ]:


X_train, X_test,y_train,y_test= train_test_split(X,y,test_size=0.3,random_state=3)


# In[ ]:


print('train set: {}  | test set: {}'.format(round(((len(y_train)*1.0)/len(X)),3), round((len(y_test)*1.0)/len(X),3)))


# In[ ]:


X_train.shape, y_train.shape, y_train


# In[ ]:


def KNNClassifier(X_train, y_train, X_test,k):

    #Create KNN object with a K coefficient
    neigh = KNeighborsClassifier(n_neighbors=k)
    neigh.fit(X_train, y_train) # Fit KNN model


    Y_pred = neigh.predict(X_test)
    return Y_pred




Y_pred =KNNClassifier(X_train, y_train, X_test,4)
  

score=metrics.accuracy_score(y_test,Y_pred)
print(score)

