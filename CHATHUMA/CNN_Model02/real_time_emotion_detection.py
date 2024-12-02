import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load the trained model
model = load_model("emotion_cnn_model.h5")
print("Model loaded successfully.")

# Define categories (must match the training order)
categories = ["Angry", "Fear", "Happy", "Neutral", "Sad", "Suprise"]

# Initialize the webcam
cap = cv2.VideoCapture(0)  # 0 for default camera
if not cap.isOpened():
    print("Error: Could not access the webcam.")
    exit()

def preprocess_frame(frame):
    # Convert to grayscale
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Resize to 48x48
    resized_frame = cv2.resize(gray_frame, (48, 48))
    # Normalize pixel values
    normalized_frame = resized_frame / 255.0
    # Reshape for the model
    reshaped_frame = np.expand_dims(normalized_frame, axis=(0, -1))
    return reshaped_frame

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture frame.")
        break

    # Preprocess the frame
    processed_frame = preprocess_frame(frame)

    # Predict the emotion
    prediction = model.predict(processed_frame)
    emotion_index = np.argmax(prediction)
    emotion = categories[emotion_index]

    # Display the emotion on the frame
    cv2.putText(frame, f"Emotion: {emotion}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    # Show the frame
    cv2.imshow("Emotion Detection", frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close windows
cap.release()
cv2.destroyAllWindows()
