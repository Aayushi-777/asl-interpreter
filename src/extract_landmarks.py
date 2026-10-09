import cv2, mediapipe as mp, csv, os
mp_hands=mp.solutions.hands
hands=mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.3)
BASE_DIR="data/raw/asl_alphabet/asl_alphabet_train/asl_alphabet_train"
OUT_CSV="data/landmarks/alphabet.csv"
SUBSET_SIZE=None
def extract_landmarks(image_path):
    img=cv2.imread(image_path)
    if img is None:
        return None
    img_rgb=cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    result=hands.process(img_rgb)
    if not result.multi_hand_landmarks:
        return None
    landmarks=[]
    for lm in result.multi_hand_landmarks[0].landmark:
        landmarks.extend([lm.x, lm.y, lm.z])
    return landmarks

def build_dataset(base_dir, out_csv, subset_size=None):
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    total_images, total_detected=0, 0
    with open(out_csv, "w", newline="") as f:
        writer=csv.writer(f)
        writer.writerow([f"p{i}" for i in range(63)]+["label"])
        for label in sorted(os.listdir(base_dir)):
            class_dir=os.path.join(base_dir, label)
            images=os.listdir(class_dir)
            if subset_size:
                images=images[:subset_size]
            for img_name in images:
                total_images+=1
                lm=extract_landmarks(os.path.join(class_dir, img_name))
                if lm and len(lm)==63:
                    writer.writerow(lm+[label])
                    total_detected+=1
            print(f"{label}: {len(images)} processed")
    print(f"\nTotal: {total_detected}/{total_images} images had a detectable hand ({100*total_detected/total_images:.1f}%)")

build_dataset(BASE_DIR, OUT_CSV, subset_size=SUBSET_SIZE)