from flask import Flask, render_template, request, jsonify, Response, redirect, url_for
import cv2
import face_recognition
import dlib
import numpy as np
import time
import mediapipe as mp
from collections import deque
from tensorflow.keras.models import load_model
import uuid
from threading import Lock
from werkzeug.security import generate_password_hash, check_password_hash
from pymongo import MongoClient
import os

app = Flask(__name__)

# Thread-safe storage for user data
user_storage = {}
storage_lock = Lock()

# MongoDB connection
client = MongoClient("mongodb+srv://mkhan331:4123Aeraf@cluster0.64ar327.mongodb.net/")
db = client['face_db']
collection = db['faces']

# Load dlib's 68-face-landmark predictor
predictor_path = "shape_predictor_68_face_landmarks.dat"
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)

# Load the trained gesture model
gesture_model = load_model("final_gesture_action_model.h5")
class_names = ["OK", "PALM_IN", "PALM_OUT", "THUMBS_DOWN", "THUMBS_UP"]

# Initialize Mediapipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Function definitions remain the same (is_valid_face, GetFaceEmbeddings, preprocess_landmarks, etc.)

def is_valid_face(frame, face_location):
    (top, right, bottom, left) = face_location
    rect = dlib.rectangle(left, top, right, bottom)
    shape = predictor(frame, rect)
    return shape.num_parts == 68

def GetFaceEmbeddings(image_data):
    np_arr = np.frombuffer(image_data, np.uint8)
    frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    face_locations = face_recognition.face_locations(rgb_frame)
    for face_location in face_locations:
        if is_valid_face(rgb_frame, face_location):
            embeddings = face_recognition.face_encodings(rgb_frame, [face_location])[0]
            return embeddings  # Keep as numpy array for comparison
    return None

def preprocess_landmarks(landmarks):
    flattened_landmarks = np.array(landmarks).flatten()  # Return 63 features
    return flattened_landmarks

def smooth_landmarks(current_landmarks, landmark_history):
    landmark_history.append(current_landmarks)
    return np.mean(np.array(landmark_history), axis=0)

def fingers_up(landmarks, handedness):
    fingers = []

    # Thumb
    if handedness == "Right":
        fingers.append(1 if landmarks[4][0] > landmarks[3][0] else 0)
    else:
        fingers.append(1 if landmarks[4][0] < landmarks[3][0] else 0)

    # Other fingers
    for tip, pip in [(8, 6), (12, 10), (16, 14), (20, 18)]:
        fingers.append(1 if landmarks[tip][1] < landmarks[pip][1] else 0)

    return str(sum(fingers))  # Return as string for consistency

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        # Determine which button was clicked
        if 'register' in request.form:
            # Registration process
            username = request.form.get('username')
            password = request.form.get('password')

            # Check if username already exists
            if collection.find_one({'username': username}):
                return render_template('login.html', message="Username already exists.")

            # Generate a unique session_id
            session_id = str(uuid.uuid4())

            # Hash the password
            hashed_password = generate_password_hash(password)

            # Store username and hashed password in user_storage
            with storage_lock:
                user_storage[session_id] = {
                    'username': username,
                    'password': hashed_password,
                    'face_embeddings': None,
                    'gesture_password': None,
                    'mode': 'register'
                }

            # Redirect to index page to get face embeddings, passing session_id
            return redirect(url_for('index', session_id=session_id))
        elif 'validate' in request.form:
            # Validation process
            username = request.form.get('username')
            password = request.form.get('password')

            # Fetch user from MongoDB
            user_document = collection.find_one({'username': username})
            if user_document and check_password_hash(user_document['password'], password):
                # Credentials are valid
                session_id = str(uuid.uuid4())

                # Store user data in user_storage
                with storage_lock:
                    user_storage[session_id] = {
                        'username': username,
                        'password': user_document['password'],  # Hashed password
                        'face_embeddings': user_document['face_embeddings'],
                        'gesture_password': user_document['gesture_password'],
                        'mode': 'validate'
                    }

                # Redirect to validation page
                return redirect(url_for('validate', session_id=session_id))
            else:
                # Invalid credentials
                return render_template('login.html', message="Invalid username or password.")
    else:
        # Render the login form
        return render_template('login.html')

@app.route('/')
def index():
    session_id = request.args.get('session_id')
    if not session_id:
        # If no session_id, redirect to login page
        return redirect(url_for('login'))
    return render_template('index.html', session_id=session_id)

@app.route('/get-face-embeddings', methods=['POST'])
def get_face_embeddings():
    session_id = request.form.get('session_id')
    if not session_id:
        return jsonify({'success': False, 'message': 'Session ID is missing.'})

    if 'image' not in request.files:
        return jsonify({'success': False, 'message': 'No image file provided.'})

    file = request.files['image']
    embeddings = GetFaceEmbeddings(file.read())
    if embeddings is not None:
        # Store embeddings in user_storage
        with storage_lock:
            if session_id in user_storage:
                user_storage[session_id]['face_embeddings'] = embeddings.tolist()  # Convert to list for JSON serialization
            else:
                # Session ID not found
                return jsonify({'success': False, 'message': 'Invalid session ID.'})
        return jsonify({'success': True, 'message': 'Embeddings generated.', 'session_id': session_id})
    else:
        return jsonify({'success': False, 'message': 'No valid face detected.'})

@app.route('/gesture')
def gesture():
    session_id = request.args.get('session_id')
    if not session_id:
        return redirect(url_for('login'))
    return render_template('gesture.html', session_id=session_id)

@app.route('/validate')
def validate():
    session_id = request.args.get('session_id')
    if not session_id:
        return redirect(url_for('login'))
    return render_template('validate.html', session_id=session_id)

@app.route("/video_feed/<session_id>")
def video_feed(session_id):
    mode = user_storage.get(session_id, {}).get('mode')
    if mode == 'register':
        # Registration mode
        return Response(generate_gesture(session_id, store_password=True), mimetype="multipart/x-mixed-replace; boundary=frame")
    elif mode == 'validate':
        # Validation mode
        return Response(generate_gesture(session_id, store_password=False), mimetype="multipart/x-mixed-replace; boundary=frame")
    else:
        return Response("Invalid session", mimetype="text/plain")

def generate_gesture(session_id, store_password):
    cap = cv2.VideoCapture(0)
    password = []
    wait_for_next_input = 5  # Time in seconds between gestures
    last_input_time = time.time()
    landmark_history = deque(maxlen=5)
    hands = mp_hands.Hands(min_detection_confidence=0.8, min_tracking_confidence=0.8)

    # For validation, get the stored gesture password
    stored_gesture_password = None
    if not store_password:
        with storage_lock:
            stored_gesture_password = user_storage.get(session_id, {}).get('gesture_password', [])

    while len(password) < 6:  # One action, four numbers, and one closing action
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)

        # Removed the ROI box drawing code

        if results.multi_handedness and results.multi_hand_landmarks:
            for hand_handedness, hand_landmarks in zip(results.multi_handedness, results.multi_hand_landmarks):
                handedness = hand_handedness.classification[0].label  # "Left" or "Right"

                # Draw landmarks
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                landmarks = [[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark]
                smoothed_landmarks = smooth_landmarks(landmarks, landmark_history)

                # Cooldown mechanism to wait for next input
                if time.time() - last_input_time >= wait_for_next_input:
                    if handedness == "Right":
                        # Use the right hand for numeric input
                        detected_fingers = fingers_up(smoothed_landmarks, handedness)
                        password.append(detected_fingers)
                    elif handedness == "Left":
                        # Use the left hand for gesture recognition
                        input_data = preprocess_landmarks(smoothed_landmarks)
                        input_data = np.expand_dims(input_data, axis=0)  # Add batch dimension

                        predictions = gesture_model.predict(input_data)
                        predicted_class = np.argmax(predictions, axis=1)[0]

                        if 0 <= predicted_class < len(class_names):
                            gesture_name = class_names[predicted_class]
                            password.append(gesture_name)

                    last_input_time = time.time()  # Reset cooldown timer

        # Display password progress
        progress = ' '.join(password)
        cv2.putText(frame, f"Password: {progress}", (10, frame.shape[0] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        _, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

        if len(password) >= 6:
            break

    cap.release()
    hands.close()

    with storage_lock:
        if session_id in user_storage:
            if store_password:
                # Registration mode: store the gesture password
                user_storage[session_id]['gesture_password'] = password
            else:
                # Validation mode: compare the password
                if password == stored_gesture_password:
                    user_storage[session_id]['gesture_match'] = True
                else:
                    user_storage[session_id]['gesture_match'] = False


@app.route("/get_password/<session_id>", methods=["GET"])
def get_password(session_id):
    with storage_lock:
        user_data = user_storage.get(session_id, {})
        password = user_data.get('gesture_password', [])
        username = user_data.get('username')
        hashed_password = user_data.get('password')
        face_embeddings = user_data.get('face_embeddings')
        mode = user_data.get('mode')

    if mode == 'register':
        if username and hashed_password and face_embeddings and password:
            # Store everything in MongoDB
            user_document = {
                'username': username,
                'password': hashed_password,
                'face_embeddings': face_embeddings,
                'gesture_password': password
            }
            try:
                collection.insert_one(user_document)
                message = "Data stored in MongoDB."
            except Exception as e:
                message = f"Error storing data in MongoDB: {str(e)}"
            return jsonify({"password": password, "message": message})
        else:
            return jsonify({"password": password, "message": "Data incomplete. Not stored in MongoDB."})
    elif mode == 'validate':
        # Return whether the gesture password matched
        gesture_match = user_data.get('gesture_match')
        if gesture_match is not None:
            return jsonify({"gesture_match": gesture_match})
        else:
            # Gesture password not yet evaluated
            return jsonify({})
    else:
        return jsonify({"message": "Invalid session."})

@app.route('/validate_face', methods=['POST'])
def validate_face():
    session_id = request.form.get('session_id')
    if not session_id:
        return jsonify({'success': False, 'message': 'Session ID is missing.'})

    if 'image' not in request.files:
        return jsonify({'success': False, 'message': 'No image file provided.'})

    with storage_lock:
        user_data = user_storage.get(session_id, {})
        stored_embeddings = np.array(user_data.get('face_embeddings'))

    if stored_embeddings is None:
        return jsonify({'success': False, 'message': 'No stored face embeddings to compare.'})

    file = request.files['image']
    embeddings = GetFaceEmbeddings(file.read())
    if embeddings is not None:
        # Compare embeddings
        distance = np.linalg.norm(embeddings - stored_embeddings)
        threshold = 0.6  # Adjust threshold as needed
        if distance < threshold:
            return jsonify({'success': True, 'message': 'Face verified successfully.'})
        else:
            return jsonify({'success': False, 'message': 'Face does not match.'})
    else:
        return jsonify({'success': False, 'message': 'No valid face detected.'})

@app.route('/face_scan')
def face_scan():
    session_id = request.args.get('session_id')
    if not session_id:
        return redirect(url_for('login'))
    return render_template('face_scan.html', session_id=session_id)


if __name__ == '__main__':
    app.run(debug=True)
