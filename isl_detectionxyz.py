import cv2
import mediapipe as mp
import copy
import itertools
import numpy as np
import pandas as pd
import string
from tensorflow import keras
from collections import deque
import pyttsx3
import threading  # Run speech in parallel

# Load the saved model
model = keras.models.load_model("model12.h5")

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

# Define alphabet (Mapping index -> label)
alphabet = ['1', '2', '3', '4', '5', '6', '7', '8', '9'] + list(string.ascii_uppercase)

# Smoothing buffer for stable predictions
prediction_buffer = deque(maxlen=10)  # Stores last 10 predictions

# Initialize text-to-speech engine
engine = pyttsx3.init()
engine.setProperty('rate', 150)  # Adjust speech speed
last_spoken = None  # Track last spoken letter
speech_thread = None  # Track speech thread

# Function to extract hand landmarks (x, y, z)
def calc_landmark_list(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]
    return [[
        min(int(l.x * image_width), image_width - 1),  # X coordinate
        min(int(l.y * image_height), image_height - 1),  # Y coordinate
        l.z  # Z coordinate (Depth)
    ] for l in landmarks.landmark]

# Function to normalize landmarks (x, y, z)
def pre_process_landmark(landmark_list):
    base_x, base_y, base_z = landmark_list[0]  # Use wrist as reference
    temp_landmark_list = [[x - base_x, y - base_y, z - base_z] for x, y, z in landmark_list]
    
    temp_landmark_list = list(itertools.chain.from_iterable(temp_landmark_list))  # Flatten list
    max_value = max(map(abs, temp_landmark_list)) if temp_landmark_list else 1  # Normalize

    return [n / max_value for n in temp_landmark_list]

# Function to run text-to-speech in a separate thread
def speak_text(text):
    global speech_thread
    if speech_thread and speech_thread.is_alive():
        return  # Avoid speaking over an ongoing speech
    speech_thread = threading.Thread(target=lambda: (engine.say(text), engine.runAndWait()))
    speech_thread.start()

# Initialize webcam
cap = cv2.VideoCapture(0)

with mp_hands.Hands(
    model_complexity=0,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7) as hands:  # Increased confidence thresholds

    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Ignoring empty frame.")
            continue
        
        image = cv2.flip(image, 1)  # Flip for selfie view
        image.flags.writeable = False
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = hands.process(image)

        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        debug_image = copy.deepcopy(image)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                landmark_list = calc_landmark_list(debug_image, hand_landmarks)
                pre_processed_landmark_list = pre_process_landmark(landmark_list)

                # Draw hand landmarks
                mp_drawing.draw_landmarks(
                    image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                df = pd.DataFrame([pre_processed_landmark_list])  # Model input
                predictions = model.predict(df, verbose=0)
                
                confidence = np.max(predictions)
                predicted_class = np.argmax(predictions)

                if confidence > 0.75:  # Ignore low-confidence predictions
                    prediction_buffer.append(alphabet[predicted_class])

                if len(prediction_buffer) > 0:
                    final_prediction = max(set(prediction_buffer), key=prediction_buffer.count)  # Most frequent value
                    cv2.putText(image, final_prediction, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 2)

                    # Speak only if the letter changes
                    if final_prediction != last_spoken:
                        last_spoken = final_prediction  # Update last spoken letter
                        speak_text(final_prediction)  # Update last spoken letter

                    print(final_prediction)

        cv2.imshow('Indian Sign Language Detector', image)
        if cv2.waitKey(1) & 0xFF == 27:  # Reduce lag by using `waitKey(1)`
            break

cap.release()
cv2.destroyAllWindows()
