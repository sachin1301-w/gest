import cv2
import mediapipe as mp
import joblib
import numpy as np
import time

model = joblib.load("gesture_model.pkl")

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1)
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

# -----------------------------
gesture_to_word = {
    "OPEN_PALM": "HOW",
    "PEACE": "ARE",
    "CLOSE_PALM": "YOU",
    "THUMBS_UP": "FINE"
}

sentence = []
last_gesture = None
last_time = 0
cooldown = 1.5  # seconds
# -----------------------------

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    detected_gesture = None

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            landmarks = []
            for lm in hand_landmarks.landmark:
                landmarks.extend([lm.x, lm.y, lm.z])

            landmarks = np.array(landmarks)

            # Normalize (same as training)
            wrist_x, wrist_y, wrist_z = landmarks[0], landmarks[1], landmarks[2]
            landmarks[0::3] -= wrist_x
            landmarks[1::3] -= wrist_y
            landmarks[2::3] -= wrist_z
            landmarks /= np.max(np.abs(landmarks))

            detected_gesture = model.predict([landmarks])[0]

            mp_draw.draw_landmarks(
                frame, hand_landmarks, mp_hands.HAND_CONNECTIONS
            )

    # -----------------------------
    # Sentence Logic
    # -----------------------------
    current_time = time.time()

    if detected_gesture and detected_gesture != last_gesture:
        if current_time - last_time > cooldown:
            word = gesture_to_word.get(detected_gesture)
            if word:
                sentence.append(word)
            last_time = current_time
            last_gesture = detected_gesture

    # Auto sentence formatting
    display_sentence = " ".join(sentence)

    if display_sentence == "HOW ARE YOU":
        display_sentence += " ?"

    if display_sentence == "FINE":
        display_sentence = "I AM FINE"

    # -----------------------------
    cv2.putText(frame, f"Gesture: {detected_gesture}",
                (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

    cv2.putText(frame, display_sentence,
                (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2)

    cv2.imshow("Gesture to Sentence", frame)

    # Reset sentence
    if cv2.waitKey(1) & 0xFF == ord('c'):
        sentence = []

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
