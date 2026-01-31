import cv2
import mediapipe as mp
import csv
import os

# ----------------------------
# YOUR REQUIRED WORDS (CHANGED HERE)
GESTURES = [
    "HI",
    "HOW_ARE_YOU",
    "I_AM_FINE",
    "WHAT_IS_YOUR_NAME",
    "MY_NAME_IS_SACHIN"
]

SAMPLES_PER_GESTURE = 200
CSV_PATH = "hand_gesture_dataset.csv"
# ----------------------------

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

file_exists = os.path.isfile(CSV_PATH)

csv_file = open(CSV_PATH, "a", newline="")
writer = csv.writer(csv_file)

# Create CSV header once
if not file_exists:
    header = []
    for i in range(21):
        header += [f"x{i}", f"y{i}", f"z{i}"]
    header.append("label")
    writer.writerow(header)

print("Press Q to quit")

for gesture in GESTURES:
    count = 0
    print(f"Collecting data for: {gesture}")

    while cap.isOpened() and count < SAMPLES_PER_GESTURE:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb)

        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                row = []
                for lm in hand_landmarks.landmark:
                    row.extend([lm.x, lm.y, lm.z])
                row.append(gesture)
                writer.writerow(row)
                count += 1

                mp_draw.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS
                )

        cv2.putText(frame, f"Gesture: {gesture}", (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.putText(frame, f"Samples: {count}/{SAMPLES_PER_GESTURE}",
                    (10, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow("Collecting Data", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            cap.release()
            csv_file.close()
            cv2.destroyAllWindows()
            exit()

cap.release()
csv_file.close()
cv2.destroyAllWindows()

print("✅ Data collection complete!")
