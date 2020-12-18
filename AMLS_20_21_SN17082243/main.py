# Viasualisation of the data labels

import pandas as pd

## Celeba labels2.csv data

celeba_df = pd.read_csv('./datasets/celeba/labels2.csv')
celeba_df = celeba_df.drop('Unnamed: 0', axis = 1)
celeba_df.head()

celeba_df.info()

celeba_df.value_counts('gender')

celeba_df.value_counts('smiling')

## Cartoon_set labels.scv data

cartoon_df = pd.read_csv('./datasets/cartoon_set/labels.csv')
cartoon_df = cartoon_df.drop('Unnamed: 0', axis = 1)
cartoon_df.head()

cartoon_df.info()

cartoon_df.isnull().sum()

cartoon_df.value_counts('face_shape')

cartoon_df.value_counts('eye_color')

# Import Required Libraries

# Task A1

import
# Task A2

# Task B1

# Task B2

