import cv2
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers

# Load the trained model

model = keras.models.load_model('digit_recognition_model.h5') # Load your trained model here

# Define a mapping from digits to textual representations
digit_to_text = {
    0: "zero",
    1: "one",
    2: "two",
    3: "three",
    4: "four",
    5: "five",
    6: "six",
    7: "seven",
    8: "eight",
    9: "nine"
}

# Function to preprocess image for the model
def preprocess_image(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img, (28, 28))
    img = img.reshape((1, 28, 28, 1)).astype("float32") / 255
    return img

# Access the camera
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Preprocess the frame
    preprocessed_frame = preprocess_image(frame)

    # Predict digit
    prediction = model.predict(preprocessed_frame)
    predicted_label = np.argmax(prediction)

    # Convert numerical prediction to textual representation
    text_label = digit_to_text[predicted_label]

    # Display the result on the frame
    cv2.putText(frame, text_label, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    # Display the frame
    cv2.imshow('Digit Recognition', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close windows
cap.release()
cv2.destroyAllWindows()
