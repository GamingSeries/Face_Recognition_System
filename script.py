import os
from PIL import Image
import numpy as np
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

# define the directory where you extracted the LFW dataset
dataset_dir = 'lfw-deepfunneled'

# we'll store the face images in X and the labels in y
X = []
y = []

# iterate through each sub-directory in the dataset directory
for class_dir in os.listdir(dataset_dir):
    if class_dir == '.DS_Store':
        continue

    # iterate through each image in the sub-directory
    for image_filename in os.listdir(os.path.join(dataset_dir, class_dir)):
        # open the image file
        img = Image.open(os.path.join(dataset_dir, class_dir, image_filename))

        # convert the image to grayscale, resize it to a common size, and convert to numpy array
        img = img.convert('L').resize((62, 47))
        img_data = np.array(img)

        # add the image data and label to our datasets
        X.append(img_data)
        y.append(class_dir)

# convert X and y to numpy arrays
X = np.array(X)
y = np.array(y)

# Split the data into a training set and a test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a CNN
model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(62, 47, 1)))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dense(len(np.unique(y))))

# Compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Reshape the data to fit the model
X_train = X_train.reshape((X_train.shape[0], 62, 47, 1))
X_test = X_test.reshape((X_test.shape[0], 62, 47, 1))

# Train the model
model.fit(X_train, y_train, epochs=10, batch_size=64,
          validation_data=(X_test, y_test))

# Evaluate the model
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=2)

print('\nTest accuracy:', test_acc)
