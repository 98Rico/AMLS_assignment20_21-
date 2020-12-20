# AMLS_assignment20_21-

In each folder A1 and A2 there a three subfolders corresponding to each extraction method used: Grayscale feature, Edges features and Landmarks Features. 
In folder B1 there is one folder for the landmark appraoch (Linear SVC and KNN Classifier) and a CNN model. In addition, there is the code to separate the data properly.
In folder B2 there is only a CNN model. In addition, there is the code to separate the data properly.


You need to download the shape predictory file at the following adress: http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2 and place in the AMLS_assignment20_21- file.

These folders contains all the algorithms tested for this project. However, in the main.py file only the methods with a best accuracy were kept. For task B1 and B2, two different files were created as these codes were run on google collad to benefit grom google's GPU.

While running main.py you need to make sure the directories fit. 

The following instructions need to be done twice (one for B1 and one for B2)

To run the CNN-based algoritms, it is required in the datasets folder to create three folder.

In the dataset we need to have the following folders:

-initial (contain the 10,000 images)

-initial_categories (contains 5 subfolder):
                    -face_1 or eye_1
                    -face_2 or eye_2
                    -face_3 or eye_3
                    -face_4 or eye_4
                    -face_5 or eye_5
                 
-final (empty)

Make sure the directories fit.

