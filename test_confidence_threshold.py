# test_confidence_threshold.py
import cv2, mediapipe as mp, os

BASE_DIR = "data/raw/asl_alphabet/asl_alphabet_train/asl_alphabet_train"

for conf in [0.5, 0.3, 0.2]:
    hands = mp.solutions.hands.Hands(static_image_mode=True, max_num_hands=1,
                                      min_detection_confidence=conf)
    for label in ["M", "N", "del"]:
        class_dir = os.path.join(BASE_DIR, label)
        images = os.listdir(class_dir)[:50]
        detected = 0
        for img_name in images:
            img = cv2.imread(os.path.join(class_dir, img_name))
            result = hands.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            if result.multi_hand_landmarks:
                detected += 1
        print(f"confidence={conf} | {label}: {detected}/50")
    hands.close()