def getGestureEmbeddings():
    import cv2
    import numpy as np
    import time
    from collections import deque
    from tensorflow.keras.models import load_model
    import mediapipe as mp

    # Load the trained gesture model
    model = load_model("final_gesture_action_model.keras")

    # Define the gesture class names
    class_names = ["OK", "PALM_IN", "PALM_OUT", "THUMBS_DOWN", "THUMBS_UP"]

    # Initialize Mediapipe Hands
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    hands = mp_hands.Hands(min_detection_confidence=0.8, min_tracking_confidence=0.8)

    # Initialize deque for smoothing landmarks
    landmark_history = deque(maxlen=5)

    # Initialize ROI Box
    cap = cv2.VideoCapture(0)
    frame_width, frame_height = int(cap.get(3)), int(cap.get(4))
    ROI_WIDTH, ROI_HEIGHT = 300, 300
    ROI_TOP_LEFT = ((frame_width - ROI_WIDTH) // 2, (frame_height - ROI_HEIGHT) // 2)
    ROI_BOTTOM_RIGHT = (ROI_TOP_LEFT[0] + ROI_WIDTH, ROI_TOP_LEFT[1] + ROI_HEIGHT)

    # Define gesture password buffer
    password = []
    wait_for_next_input = 1.5  # Time in seconds between gestures
    last_input_time = time.time()

    # Function to preprocess landmarks for gesture recognition
    def preprocess_landmarks(landmarks):
        flattened_landmarks = np.array(landmarks).flatten()  # Return 63 features
        return flattened_landmarks

    # Smooth landmarks for numeric passcode detection
    def smooth_landmarks(current_landmarks):
        landmark_history.append(current_landmarks)
        return np.mean(np.array(landmark_history), axis=0)

    # Logic to count raised fingers
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

    # Capture user input
    while len(password) < 6:  # One action, four numbers, and one closing action
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)

        # Draw centered ROI box
        cv2.rectangle(frame, ROI_TOP_LEFT, ROI_BOTTOM_RIGHT, (0, 255, 0), 2)
        cv2.putText(frame, "Set Your Gesture Password", (ROI_TOP_LEFT[0], ROI_TOP_LEFT[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        if results.multi_handedness and results.multi_hand_landmarks:
            for hand_handedness, hand_landmarks in zip(results.multi_handedness, results.multi_hand_landmarks):
                handedness = hand_handedness.classification[0].label  # "Left" or "Right"

                # Draw landmarks
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                landmarks = [[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark]
                smoothed_landmarks = smooth_landmarks(landmarks)

                # Cooldown mechanism to wait for next input
                if time.time() - last_input_time >= wait_for_next_input:
                    if handedness == "Right":
                        # Use the right hand for numeric input
                        detected_fingers = fingers_up(smoothed_landmarks, handedness)
                        cv2.putText(frame, f"Right Hand: {detected_fingers} Fingers", (10, 60),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                        password.append(detected_fingers)
                    elif handedness == "Left":
                        # Use the left hand for gesture recognition
                        input_data = preprocess_landmarks(smoothed_landmarks)
                        input_data = np.expand_dims(input_data, axis=0)  # Add batch dimension

                        predictions = model.predict(input_data)
                        predicted_class = np.argmax(predictions, axis=1)[0]

                        if 0 <= predicted_class < len(class_names):
                            gesture_name = class_names[predicted_class]
                            cv2.putText(frame, f"Gesture: {gesture_name}", (10, 100),
                                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                            password.append(gesture_name)

                    last_input_time = time.time()  # Reset cooldown timer

        # Display current password progress
        progress = ' '.join(password)
        cv2.putText(frame, f"Password: {progress}", (10, ROI_BOTTOM_RIGHT[1] + 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        cv2.imshow("Gesture Password Setup", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    return password

password = getGestureEmbeddings()
print("Your password is:", password)

