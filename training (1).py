'''
This folder consist of train-folder---> consists of 16000 images of 16 classes,
                       validation-folder ---> consists of 900 images for validation and no class lables availiable
                       train_lables.csv ---> contains id of image to class lables only for training data
                       training.py ---> contains code for training models and saving them
                       prediction.py ---> contains code for loading testing data, saved models and predicting output/class labels
'''

# Importing packages
import cv2
import tensorflow
import os
import numpy as np
import pandas as pd
from tensorflow.keras.utils import to_categorical




# Loading training images and their lables and converting images data to numpy arrays
train_data = []
train_labels = []
training_dir = os.path.join(os.getcwd(),'train')
for image in os.listdir(training_dir):
    img = cv2.imread(os.path.join(training_dir,image))#,cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img,(256,256))
    train_data.append(np.array(img))
    # Same images with repeated name might be availiable in dataset.
    # Checking for valid,non-repeated images with same name and loading them only
    try:
        train_labels.append(int(image.split(".")[0]))
    except:
        del train_data[-1]
train_data = np.array(train_data)
print("\nCompleted collecting images\n")
print("Shape of training data:",train_data.shape)




# Loading train_labels.csv which contains labels and corresponding output class
classes = pd.read_csv(os.path.join(os.getcwd(),'train_labels.csv'))
# Loading class labels corresponding to labels of images for training.
train_classes = []
for i in train_labels:
    train_classes.append(classes.iloc[i,1])
train_classes = np.array(train_classes)
encoded_train_classes = to_categorical(train_classes)
print("\n\n")




# Creating a model using combination of DL and ML algorithms(deep hybrid learning)
from tensorflow.keras import Input,Model
from tensorflow.keras.layers import Conv2D,MaxPooling2D,BatchNormalization,Flatten,Dense,Concatenate
from tensorflow.keras.utils import plot_model
from matplotlib import pyplot as plt


x = Input(shape=train_data.shape[1:])
# top model
top_model = Conv2D(32,(3,3),activation='relu',data_format="channels_last",input_shape=train_data.shape[1:])(x)
top_model = MaxPooling2D(pool_size=(3,3))(top_model)
top_model = BatchNormalization()(top_model)
top_model = Conv2D(64,(3,3),activation='relu')(top_model)
top_model = MaxPooling2D(pool_size=(3,3))(top_model)
top_model = BatchNormalization()(top_model)

# Dividing features into 4 subsets, that divides each image/filter into 4 equal parts
top_left_input = top_model[:,0:int(top_model.shape[1]/2),0:int(top_model.shape[2]/2),:] # Top_left
bottom_left_input = top_model[:,int(top_model.shape[1]/2):,0:int(top_model.shape[2]/2),:]  # Bottom_left
top_right_input = top_model[:,0:int(top_model.shape[1]/2),int(top_model.shape[2]/2):,:]  # Top_right
bottom_right_input = top_model[:,int(top_model.shape[1]/2):,int(top_model.shape[2]/2):,:]   # Bottom_right

# top_left model, to apply CNN on top_left part of image/filter
top_left_model = Conv2D(128,(2,2),activation='relu',data_format="channels_last",input_shape=top_left_input.shape[1:])(top_left_input)
top_left_model = MaxPooling2D(pool_size=(2,2))(top_left_model)
top_left_model = BatchNormalization()(top_left_model)
top_left_model = Conv2D(256,(2,2),activation='relu')(top_left_model)
top_left_model = MaxPooling2D(pool_size=(2,2))(top_left_model)
top_left_model = BatchNormalization()(top_left_model)
top_left_model = Flatten()(top_left_model)
top_left_model = Dense(64,activation='relu')(top_left_model)
top_left_model = Dense(32,activation='relu')(top_left_model)
top_left_model = Dense(16,activation='softmax')(top_left_model)

# bottom_left model, to apply CNN on bottom_left part of image/filter
bottom_left_model = Conv2D(128,(2,2),activation='relu',data_format="channels_last",input_shape=bottom_left_input.shape[1:])(bottom_left_input)
bottom_left_model = MaxPooling2D(pool_size=(2,2))(bottom_left_model)
bottom_left_model = BatchNormalization()(bottom_left_model)
bottom_left_model = Conv2D(256,(2,2),activation='relu')(bottom_left_model)
bottom_left_model = MaxPooling2D(pool_size=(2,2))(bottom_left_model)
bottom_left_model = BatchNormalization()(bottom_left_model)
bottom_left_model = Flatten()(bottom_left_model)
bottom_left_model = Dense(64,activation='relu')(bottom_left_model)
bottom_left_model = Dense(32,activation='relu')(bottom_left_model)
bottom_left_model = Dense(16,activation='softmax')(bottom_left_model)

# top_right model, to apply CNN on top_right part of image/filter
top_right_model = Conv2D(128,(2,2),activation='relu',data_format="channels_last",input_shape=top_right_input.shape[1:])(top_right_input)
top_right_model = MaxPooling2D(pool_size=(2,2))(top_right_model)
top_right_model = BatchNormalization()(top_right_model)
top_right_model = Conv2D(256,(2,2),activation='relu')(top_right_model)
top_right_model = MaxPooling2D(pool_size=(2,2))(top_right_model)
top_right_model = BatchNormalization()(top_right_model)
top_right_model = Flatten()(top_right_model)
top_right_model = Dense(64,activation='relu')(top_right_model)
top_right_model = Dense(32,activation='relu')(top_right_model)
top_right_model = Dense(16,activation='softmax')(top_right_model)

# bottom_right model, to apply CNN on bottom_right part of image/filter
bottom_right_model = Conv2D(128,(2,2),activation='relu',data_format="channels_last",input_shape=bottom_right_input.shape[1:])(bottom_right_input)
bottom_right_model = MaxPooling2D(pool_size=(2,2))(bottom_right_model)
bottom_right_model = BatchNormalization()(bottom_right_model)
bottom_right_model = Conv2D(256,(2,2),activation='relu')(bottom_right_model)
bottom_right_model = MaxPooling2D(pool_size=(2,2))(bottom_right_model)
bottom_right_model = BatchNormalization()(bottom_right_model)
bottom_right_model = Flatten()(bottom_right_model)
bottom_right_model = Dense(64,activation='relu')(bottom_right_model)
bottom_right_model = Dense(32,activation='relu')(bottom_right_model)
bottom_right_model = Dense(16,activation='softmax')(bottom_right_model)

# concatenating all the outputs of top_left_model,top_right_model,bottom_left_model,bottom_right_model
output = Concatenate()([top_left_model,bottom_left_model,top_right_model,bottom_right_model])
output = Dense(16,activation='softmax')(output)

# Building the DL-model
model = Model(inputs=x,outputs=output)

'''
# Saving the architecture of DL-model
plot_model(model,to_file='FinalModel.png',show_shapes=True)
'''

# compiling and fitting the entire model
model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['Accuracy'])

model.fit(train_data,encoded_train_classes,epochs=10,batch_size=16,validation_split=0.2,shuffle=True,use_multiprocessing=True)

# model is being saved
model.save("DL_model.h5")
print("\nDL model is saved!!!\n")




# Finally, we are using K-NN algorithm to predict final output
from sklearn.neighbors import KNeighborsClassifier

xtrain = model.predict(train_data)
ytrain = train_classes
hybrid_model_knn = KNeighborsClassifier(n_neighbors=6)

# Fitting KNN model with data
hybrid_model_knn.fit(xtrain, ytrain)

# Predicting lables for training data
ypred = hybrid_model_knn.predict(xtrain)

# Accuracy score and classification report of KNN model
from sklearn.metrics import confusion_matrix,accuracy_score,classification_report

print("\nAccuracy of model: ",accuracy_score(ytrain, ypred),"\n")
#print(classification_report(ytrain,ypred))




# Saving the model to disk
import joblib

joblib.dump(hybrid_model_knn,'Output_Model.pkl')
print("\nK-NN model is saved!!!\n")


# Accuracy of entire model with 10-epoches and batch_size=32 is 56.51875
