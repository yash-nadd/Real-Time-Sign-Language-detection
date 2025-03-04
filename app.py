from flask import Flask, render_template, Response, request
import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import string
from tensorflow import keras
from collections import deque
import pyttsx3
import threading

app = Flask(__name__)

# Load both models
models = {
    "alphabet": keras.models.load_model("models\model12.h5"),
    "words": keras.models.load_model("models\wordmodel1.h5")
}

current_model = "alphabet"

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

# Define classes
alphabet = ['1', '2', '3', '4', '5', '6', '7', '8', '9'] + list(string.ascii_uppercase)
word_list = ['afraid', 'agree', 'assistance', 'bad', 'become', 'college', 'doctor', 'from', 
             'pain', 'pray', 'secondary', 'skin', 'small', 'specific', 'stand', 'today', 
             'warn', 'which', 'work', 'you']

prediction_buffer = deque(maxlen=10)
word_buffer = deque(maxlen=15)

last_spoken = None
speech_thread = None

def speak_text(text):
    def run_tts():
        local_engine = pyttsx3.init()
        local_engine.setProperty('rate', 150)
        local_engine.say(text)
        local_engine.runAndWait()
    
    global speech_thread
    if speech_thread and speech_thread.is_alive():
        return
    speech_thread = threading.Thread(target=run_tts)
    speech_thread.start()

def calc_landmark_list(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]
    return [[
        min(int(l.x * image_width), image_width - 1),
        min(int(l.y * image_height), image_height - 1),
        l.z
    ] for l in landmarks.landmark]

def pre_process_landmark(landmark_list):
    base_x, base_y, base_z = landmark_list[0]
    temp_landmark_list = [[x - base_x, y - base_y, z - base_z] for x, y, z in landmark_list]
    temp_landmark_list = np.array(temp_landmark_list).flatten()
    max_value = max(map(abs, temp_landmark_list)) if temp_landmark_list.any() else 1
    return temp_landmark_list / max_value

def generate_frames():
    cap = cv2.VideoCapture(0)

    with mp_hands.Hands(
        model_complexity=0,
        max_num_hands=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.7) as hands:
        
        while True:
            success, image = cap.read()
            if not success:
                continue
            
            image = cv2.flip(image, 1)
            results = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            
            detected_word = ""
            
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    landmark_list = calc_landmark_list(image, hand_landmarks)
                    processed_landmark_list = pre_process_landmark(landmark_list)
                    
                    df = pd.DataFrame([processed_landmark_list])
                    model = models[current_model]
                    predictions = model(df, training=False).numpy()
                    
                    confidence = np.max(predictions)
                    predicted_class = np.argmax(predictions)
                    
                    if confidence > 0.75:
                        if current_model == "alphabet":
                            prediction_buffer.append(alphabet[predicted_class])
                        else:
                            prediction_buffer.append(word_list[predicted_class])
                    
                    if len(prediction_buffer) > 0:
                        final_prediction = max(set(prediction_buffer), key=prediction_buffer.count)
                        word_buffer.append(final_prediction)
                        
                        if len(set(word_buffer)) == 1:
                            stable_word = word_buffer[0]
                            
                            global last_spoken
                            if stable_word != last_spoken:
                                last_spoken = stable_word
                                speak_text(stable_word)
                            detected_word = stable_word
                    
                    mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            if detected_word:
                cv2.putText(image, detected_word, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 2)
            
            ret, buffer = cv2.imencode('.jpg', image)
            frame = buffer.tobytes()
            yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()

@app.route('/')
def index():
    return render_template('index.html', model=current_model)

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/switch_model', methods=['POST'])
def switch_model():
    global current_model
    current_model = request.form.get("model", "alphabet")
    return "", 204

if __name__ == '__main__':
    app.run(debug=True, threaded=True)
