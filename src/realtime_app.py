import cv2, mediapipe as mp, numpy as np, pyttsx3, pickle
from tensorflow.keras.models import load_model
from collections import deque, Counter

model=load_model("models/best_model.keras")
with open("models/label_encoder.pkl", "rb") as f:
    label_encoder=pickle.load(f)
print("Model loaded. Classes", list(label_encoder.classes_))

mp_hands=mp.solutions.hands
mp_draw=mp.solutions.drawing_utils
hands=mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.3,
    min_tracking_confidence=0.5,
)

engine=pyttsx3.init()
engine.setProperty("rate", 150)

prediction_buffer=deque(maxlen=15)
CONF_THRESHOLD=0.75
last_spoken=None

def normalize_landmarks(landmarks):
    coords=np.array(landmarks).reshape(-1, 3)
    coords-=coords[0]
    max_val=np.max(np.linalg.norm(coords, axis=1))
    coords/=(max_val+1e-6)
    return coords.flatten()

cap=cv2.VideoCapture(0)
if not cap.isOpened():
    print("ERROR: Could not open webcam.")
    exit()
print("Starting webcam feed. Press 'q' to quit, 's' to speak the current letter.")
while True:
    ret, frame=cap.read()
    if not ret:
        break
    frame=cv2.flip(frame, 1)
    rgb=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result=hands.process(rgb)
    stable_label=""
    confidence=0.0
    if result.multi_hand_landmarks:
        hand_landmarks=result.multi_hand_landmarks[0]
        mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        raw=[]
        for lm in hand_landmarks.landmark:
            raw.extend([lm.x, lm.y, lm.z])

        features=normalize_landmarks(raw).reshape(1, -1)
        probs=model.predict(features, verbose=0)[0]
        confidence=float(np.max(probs))
        pred_idx=int(np.argmax(probs))
        top3_idx = np.argsort(probs)[-3:][::-1]
        top3 = [(label_encoder.inverse_transform([i])[0], round(float(probs[i]), 3)) for i in top3_idx]
        print("Top 3:", top3)
        if confidence>=CONF_THRESHOLD:
            pred_label=label_encoder.inverse_transform([pred_idx])[0]
            prediction_buffer.append(pred_label)
        if prediction_buffer:
            stable_label, count=Counter(prediction_buffer).most_common(1)[0]
            if count/len(prediction_buffer)<0.6:
                stable_label=""
    else:
        prediction_buffer.clear()

    cv2.rectangle(frame, (0, 0), (400, 90), (0, 0, 0), -1)
    if stable_label:
        cv2.putText(frame, f"Sign: {stable_label}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
        cv2.putText(frame, f"Confidence: {confidence:.2f}", (10, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
    else:
        cv2.putText(frame, f"No sign detected", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 2)
    cv2.imshow("ASL Interpreter-Live Demo", frame)
    key=cv2.waitKey(1) & 0xFF
    if key==ord('q'):
        break
    elif key==ord('s') and stable_label:
        engine.say(stable_label)
        engine.runAndWait()

cap.release()
cv2.destroyAllWindows()
