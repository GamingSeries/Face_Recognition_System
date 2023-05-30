import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras import utils
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from PIL import Image

path_to_dataset = 'deepfunneled'
labels = []
faces = []

for folder in os.listdir(path_to_dataset):
    for filename in os.listdir(path_to_dataset + '/' + folder):
        image_path = path_to_dataset + '/' + folder + '/' + filename
        image = Image.open(image_path).convert("L")
        image = np.array(image).astype('float32')
        image /= 255
        image = image.reshape(250, 250, 1)
        faces.append(image)
        labels.append(folder)

le = LabelEncoder()
labels = le.fit_transform(labels)

faces = np.array(faces)
labels = np.array(labels)

X_train, X_test, y_train, y_test = train_test_split(faces, labels, test_size=0.2, random_state=42)

num_classes = len(set(labels))
y_train = utils.to_categorical(y_train, num_classes)
y_test = utils.to_categorical(y_test, num_classes)

model = Sequential()

model.add(Conv2D(64, kernel_size=(3, 3), activation='relu', input_shape=(250, 250, 1), data_format='channels_last', kernel_regularizer=l2(0.01)))
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Dropout(0.5))

model.add(Conv2D(128, kernel_size=(3, 3), activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Dropout(0.5))

model.add(Conv2D(256, kernel_size=(3, 3), activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Dropout(0.5))

model.add(Flatten())
model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss=categorical_crossentropy,
              optimizer=Adam(),
              metrics=['accuracy'])

model.fit(X_train, y_train,
          batch_size=64,
          epochs=10,
          verbose=1,
          validation_data=(X_test, y_test))

accuracy = model.evaluate(X_test, y_test)[1]
print("Test Accuracy: {:.4f}".format(accuracy))

model.save('face_recognition_model.h5')