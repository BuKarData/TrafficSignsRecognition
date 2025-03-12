from google.colab import files
files.upload()

!pip install -q kaggle

!mkdir -p ~/.kaggle
!cp kaggle.json ~/.kaggle/

!chmod 600 ~/.kaggle/kaggle.json

!mkdir trafficSignDataset

# Commented out IPython magic to ensure Python compatibility.
# %cd trafficSignDataset

!kaggle datasets list -s gtsrb-german-traffic-sign

!kaggle datasets download -d meowmeowmeowmeowmeow/gtsrb-german-traffic-sign

# Commented out IPython magic to ensure Python compatibility.
# %cd ..

!unzip trafficSignDataset/gtsrb-german-traffic-sign.zip -d trafficSignDataset
!rm trafficSignDataset/gtsrb-german-traffic-sign.zip
!rm -rf trafficSignDataset/Meta
!rm -rf trafficSignDataset/meta
!rm -rf trafficSignDataset/test
!rm -rf trafficSignDataset/train
!rm trafficSignDataset/Meta.csv

#importing libraries
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as imread
import seaborn as sns
import random
from PIL import Image
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout, Conv2D, MaxPooling2D

#Visualizing data
plt.figure(figsize=(10, 10))
path = "trafficSignDataset/Test"
for i in range(1,9):
  plt.subplot(4,4,i)
  plt.tight_layout()
  rand_img = imread.imread(path + '/' + random.choice(sorted(os.listdir(path))))
  plt.imshow(rand_img)
  plt.xlabel(rand_img.shape[1], fontsize = 10)
  plt.ylabel(rand_img.shape[0], fontsize = 10)

#Mean of dimensions
dim1 = []
dim2 = []

for i in range(0,43):
  labels = 'trafficSignDataset/Train/' + '/{0}'.format(i)
  image_path = os.listdir(labels)
  for j in image_path:
    img = imread.imread(labels + '/' + j)
    dim1.append(img.shape[0])
    dim2.append(img.shape[1])

#printing mean dimension
print("Dimension 1 mean: ", np.mean(dim1), "Dimension 2 mean: ", np.mean(dim2))

images = []
label_id = []

#reshaping pictures
for i in range(43):
  labels = 'trafficSignDataset/Train/' + '/{0}'.format(i)
  image_path = os.listdir(labels)
  for j in image_path:
    img = Image.open(labels+ '/' + j)
    img = img.resize((50,50))
    img = np.array(img)
    images.append(img)
    label_id.append(i)

images = np.array(images)
#normalization
images = images/255

label_id = np.array(label_id)
label_id.shape
images.shape

label_counts = pd.DataFrame(label_id).value_counts()
label_counts.head()

#splitting data
X_train, x_val, Y_train, y_val = train_test_split(images, label_id, test_size=0.2, random_state=42)

y_train_cat = to_categorical(Y_train)
y_val_cat = to_categorical(y_val)

#building model

model = Sequential()
model.add(Conv2D(filters=64, kernel_size=(3,3), activation='relu', input_shape=X_train.shape[1:], padding = 'same'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.5))

model.add(Conv2D(filters=64, kernel_size=(3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.5))

model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(43, activation='softmax'))

model.compile(loss='sparse_categorical_crossentropy', optimizer = 'adam', metrics=['accuracy'])
model.summary()

model.fit(X_train, Y_train, batch_size=128, epochs=5, validation_data=(x_val, y_val), verbose = 2)

model_evaluation = pd.DataFrame(model.history.history)
model_evaluation[['accuracy','val_accuracy']].plot()
model_evaluation[['loss','val_loss']].plot()

test_path  = 'trafficSignDataset/Test'
!rm trafficSignDataset/Test/GT-final_test.csv

#defining scaling function
def scaling(test_images, test_path):
  images = []

  image_path = test_images

  for i in image_path:
    img = Image.open(test_path + '/' + i)
    img = img.resize((50,50))
    img = np.array(img)
    images.append(img)

  images = np.array(images)
  images = images/255 #normalization
  return images

test_images = scaling(sorted(os.listdir(test_path)), test_path)

test = pd.read_csv('trafficSignDataset/Test.csv')
y_test = test['ClassId'].values
y_test

y_pred = model.predict(test_images)
y_pred_classes = np.argmax(y_pred, axis=-1)
y_pred_classes

all_labels = []
for i in range(43):
  all_labels.append(i)

image = Image.open(test_path + '/00001.png')
image

print('Original label: ', all_labels[y_test[1]])
print('Predicted label: ', all_labels[y_pred_classes[1]])



