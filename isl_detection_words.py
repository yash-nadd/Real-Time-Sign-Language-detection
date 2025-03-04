import cv2
import mediapipe as mp
import copy
import itertools
import numpy as np
import pandas as pd
from tensorflow import keras
from collections import deque
import pyttsx3
import string
import threading  # Run speech in parallel

# **Load the new word-based model**
model = keras.models.load_model("wordmodel1.h5")  # Ensure this model is trained for word prediction

# Print model output shape for debugging
print(f"Model Output Shape: {model.output_shape}")

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

# **Define word list (Mapping index -> word)**
# Ensure this matches the number of classes in the trained model
word_list = [
    'afraid', 'agree', 'assistance', 'bad', 'become', 'college',
       'doctor', 'from', 'pain', 'pray', 'secondary', 'skin', 'small',
       'specific', 'stand', 'today', 'warn', 'which', 'work', 'you']

# Check if word_list matches model output classes
num_classes = model.output_shape[-1]
if len(word_list) != num_classes:
    print(f"⚠ Warning: Model expects {num_classes} classes but word_list has {len(word_list)} words!")

# **Smoothing buffer for stable predictions**
prediction_buffer = deque(maxlen=5)  # Stores last 5 predictions to reduce flickering

# **Initialize text-to-speech engine**
engine = pyttsx3.init()
engine.setProperty('rate', 150)  # Adjust speech speed
last_spoken = None  # Track last spoken word
speech_thread = None  # Track speech thread

# **Function to extract hand landmarks (x, y, z)**
def calc_landmark_list(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]
    return [[
        min(int(l.x * image_width), image_width - 1),  # X coordinate
        min(int(l.y * image_height), image_height - 1),  # Y coordinate
        l.z  # Z coordinate (Depth)
    ] for l in landmarks.landmark]

# **Function to normalize landmarks (x, y, z)**
def pre_process_landmark(landmark_list):
    base_x, base_y, base_z = landmark_list[0]  # Use wrist as reference
    temp_landmark_list = [[x - base_x, y - base_y, z - base_z] for x, y, z in landmark_list]
    
    temp_landmark_list = list(itertools.chain.from_iterable(temp_landmark_list))  # Flatten list
    max_value = max(map(abs, temp_landmark_list)) if temp_landmark_list else 1  # Normalize

    return [n / max_value for n in temp_landmark_list]

# **Function to run text-to-speech in a separate thread**
def speak_text(text):
    global speech_thread
    if speech_thread and speech_thread.is_alive():
        return  # Avoid speaking over an ongoing speech
    speech_thread = threading.Thread(target=lambda: (engine.say(text), engine.runAndWait()))
    speech_thread.start()

# **Initialize webcam**
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

                # **Fix: Prevent IndexError**
                if predicted_class < len(word_list):  # Check if valid index
                    prediction_buffer.append(word_list[predicted_class])
                else:
                    print(f"⚠ Warning: Predicted index {predicted_class} is out of range!")

                if len(prediction_buffer) > 0:
                    final_prediction = max(set(prediction_buffer), key=prediction_buffer.count)  # Most frequent word
                    cv2.putText(image, final_prediction, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 2)

                    # Speak only if the word changes
                    if final_prediction != last_spoken:
                        speak_text(final_prediction)  # Use threaded TTS
                        last_spoken = final_prediction  # Update last spoken word

                    print(final_prediction)

        cv2.imshow('Indian Sign Language Word Detector', image)
        if cv2.waitKey(1) & 0xFF == 27:  # Press ESC to exit
            break

cap.release()
cv2.destroyAllWindows()
