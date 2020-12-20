#!/usr/bin/env python
# coding: utf-8

# # Split Dataset

# The following code is used to split the dataset accordint to a phase and an eye category. 

# ## Import Libraries

# In[1]:


import os, random, shutil
import pandas as pd


# # Define some path starting points

# In[2]:


source = 'datasets/initial/'
intermediate = 'datasets/initial_categories/'
destination = 'datasets/final/'


# We create a path to reach each image

# In[3]:


global basedir, image_paths, target_size
basedir = './datasets/'
images_dir = os.path.join(basedir,'initial')
labels_filename = 'labels.csv'


# Then we create subfolder for the phase and subsubfolders for the eye category and create a directory

# In[4]:


labels = ['eye_1', 'eye_2', 'eye_3', 'eye_4', 'eye_5']

for phase in  ['train/', 'test/', 'val/']:
    # Creating a directory for each phase of model usage
    os.mkdir(destination + phase)

    # Creating a directory for each pill inside each model phase
    for label in labels:
        os.mkdir(destination + phase + label)


# Create a directory for the labels

# In[5]:


for label in labels:
        os.mkdir(source + label)


# In[7]:


eye_df = pd.read_csv('datasets/' + labels_filename)
eye_df = eye_df.drop('Unnamed: 0', axis = 1)
eye_df = eye_df.drop('face_shape', axis = 1)
eye_df.head()


# In[8]:


images_dir


# In[9]:


image_paths = [os.path.join(images_dir, l) for l in os.listdir(images_dir)]
image_paths


# In[10]:


target_size = None
labels_file = open(os.path.join(basedir, labels_filename), 'r')
labels_file


# In[11]:


lines = labels_file.readlines()
lines


# We only keep the image number with its eye category number as follow

# In[12]:


eye_labels = {line.split(',')[0] : int(line.split(',')[1]) for line in lines[1:]}
eye_labels


# In[13]:


eye_labels['0']


# We order the pictures to send them appropiately to their correspondong folder

# In[14]:


for key, value in eye_labels.items():
    if value == 0:
        shutil.move(source + key + '.png',intermediate + 'eye_1')
    elif value == 1:
        shutil.move(source + key + '.png',intermediate + 'eye_2')
    elif value == 2:
        shutil.move(source + key + '.png',intermediate + 'eye_3')
    elif value == 3:
        shutil.move(source + key + '.png',intermediate + 'eye_4')
    else:
        shutil.move(source + key + '.png',intermediate + 'eye_5')


# Finally we send the pictures to the final folder by ransomly placing them in the train, test or val subfolder

# In[15]:


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

