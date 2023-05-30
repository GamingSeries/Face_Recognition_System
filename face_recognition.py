import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.mobilenet import preprocess_input

# Load the model we trained
model = load_model('face_recognition_model.h5')

def prepare_image(file):
    img_path = ''
    img = image.load_img(img_path + file, target_size=(224, 224))
    img_array = img_to_array(img)
    img_array_expanded_dims = np.expand_dims(img_array, axis=0)
    return preprocess_input(img_array_expanded_dims)

# Initialize the webcam feed
cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()

    # Process image for model prediction
    face_crop = cv2.resize(frame, (224, 224))
    face_crop = img_to_array(face_crop)
    face_crop = np.expand_dims(face_crop, axis=0)
    face_crop = preprocess_input(face_crop)

    # Use model to predict face
    confidences = model.predict(face_crop)
    idx = np.argmax(confidences)

    # Add text label for predicted face
    cv2.putText(frame, str(idx), (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

    # Display the frame
    cv2.imshow('Face Recognition System - Muneeb Farooq', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
