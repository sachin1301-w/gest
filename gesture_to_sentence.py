import cv2
import mediapipe as mp
import joblib
import numpy as np
import time

# =============================
# LOAD MODEL
# =============================
model = joblib.load("gesture_model.pkl")

# =============================
# MEDIAPIPE
# =============================
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)
mp_draw = mp.solutions.drawing_utils

# =============================
# GESTURE → TEXT MAP (MATCH TRAINING)
# =============================
GESTURE_TO_TEXT = {
    "HI": "Hi",
    "HOW_ARE_YOU": "How are you?",
    "I_AM_FINE": "I am fine.",
    "WHAT_IS_YOUR_NAME": "What is your name?",
    "MY_NAME_IS_SACHIN": "My name is Sachin."
}

# =============================
# SENTENCE CONTROL
# =============================
final_sentence = ""
last_added = ""
last_time = 0
COOLDOWN = 1.5  # seconds

# =============================
# TEXT WRAP FUNCTION
# =============================
def wrap_text(text, width=35):
    words = text.split(" ")
    lines = []
    current = ""

    for word in words:
        if len(current + " " + word) <= width:
            current += " " + word
        else:
            lines.append(current.strip())
            current = word

    if current:
        lines.append(current.strip())
    return lines

# =============================
# CAMERA
# =============================
cap = cv2.VideoCapture(0)
print("✅ Gesture → Sentence system started")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    gesture_label = "NO_HAND"

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:

            # -----------------------------
            # FEATURE EXTRACTION
            # -----------------------------
            landmarks = []
            for lm in hand_landmarks.landmark:
                landmarks.extend([lm.x, lm.y, lm.z])

            landmarks = np.array(landmarks, dtype=float)

            # SAME NORMALIZATION AS TRAINING
            wrist_x, wrist_y, wrist_z = landmarks[0], landmarks[1], landmarks[2]
            landmarks[0::3] -= wrist_x
            landmarks[1::3] -= wrist_y
            landmarks[2::3] -= wrist_z

            max_val = np.max(np.abs(landmarks))
            if max_val != 0:
                landmarks /= max_val

            # -----------------------------
            # PREDICTION
            # -----------------------------
            prediction = model.predict([landmarks])[0]
            gesture_label = prediction

            current_time = time.time()

            # Sentence update (no repetition)
            if gesture_label in GESTURE_TO_TEXT:
                new_text = GESTURE_TO_TEXT[gesture_label]

                if new_text != last_added and current_time - last_time > COOLDOWN:
                    final_sentence += new_text + " "
                    last_added = new_text
                    last_time = current_time

            mp_draw.draw_landmarks(
                frame, hand_landmarks, mp_hands.HAND_CONNECTIONS
            )

    # -----------------------------
    # DISPLAY
    # -----------------------------
    cv2.putText(frame, f"Gesture: {gesture_label}",
                (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.putText(frame, "Sentence:",
                (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

    lines = wrap_text(final_sentence, width=35)
    y = 130
    for line in lines:
        cv2.putText(frame, line, (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
        y += 35

    cv2.imshow("Hand Gesture to Sentence", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
