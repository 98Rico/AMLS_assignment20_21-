#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os, random, shutil
import pandas as pd


# In[2]:


source = 'datasets/img/'
intermediate = 'datasets/initial_dataset/'
destination = 'datasets/final_dataset/'


# Let's create category folders

# In[3]:


# PATH TO ALL IMAGES
global basedir, image_paths, target_size
basedir = './datasets/'
images_dir = os.path.join(basedir,'img')
labels_filename = 'labels.csv'


# In[4]:


labels = ['face_1', 'face_2', 'face_3', 'face_4', 'face_5']

for phase in  ['train/', 'test/', 'val/']:
    # Creating a directory for each phase of model usage
    os.mkdir(destination + phase)

    # Creating a directory for each pill inside each model phase
    for label in labels:
        os.mkdir(destination + phase + label)


# In[5]:


for label in labels:
        os.mkdir(intermediate + label)


# In[6]:


face_df = pd.read_csv(basedir + labels_filename)
face_df = face_df.drop('Unnamed: 0', axis = 1)
face_df = face_df.drop('eye_color', axis = 1)
face_df.head()


# In[7]:


images_dir


# In[8]:


face_df.isnull().sum()


# In[9]:


face_df.value_counts('face_shape')


# In[10]:


image_paths = [os.path.join(images_dir, l) for l in os.listdir(images_dir)]
image_paths


# In[11]:


target_size = None
labels_file = open(os.path.join(basedir, labels_filename), 'r')
labels_file


# In[12]:


lines = labels_file.readlines()
lines


# In[13]:


face_labels = {line.split(',')[0] : int(line.split(',')[2]) for line in lines[1:]}
face_labels


# In[14]:


face_labels['0']


# In[15]:


for key, value in face_labels.items():
    if value == 0:
        shutil.move(source + key + '.png',intermediate + 'face_1')
    elif value == 1:
        shutil.move(source + key + '.png',intermediate + 'face_2')
    elif value == 2:
        shutil.move(source + key + '.png',intermediate + 'face_3')
    elif value == 3:
        shutil.move(source + key + '.png',intermediate + 'face_4')
    else:
        shutil.move(source + key + '.png',intermediate + 'face_5')


# In[16]:


for label in labels:
    files = os.listdir(intermediate + label)

    # Sorting and randomly shuffling images
    files.sort()
    random.seed(230)
    random.shuffle(files)

    # Applying separation of 70% train, 20% validation and 10% test
    split_train = int(0.7 * len(files))
    split_val = int(0.9 * len(files))

    # Separating images into train, validation and testing
    train_files = files[:split_train]
    val_files = files[split_train:split_val]
    test_files = files[split_val:]

# Moving images to each respective folder
    for f in train_files:
        shutil.move(intermediate + label + '/' + f, destination + 'train/' + label)

    for f in val_files:
        shutil.move(intermediate  + label + '/' + f, destination + 'val/' + label)

    for f in test_files:
        shutil.move(intermediate + label + '/' + f, destination + 'test/' + label)


# In[ ]:




