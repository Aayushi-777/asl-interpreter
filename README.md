# ASL Interpreter Using Image Detection

A real-time American Sign Language (ASL) alphabet recognizer that uses a standard webcam. The system detects the signer's hand with MediaPipe, converts it into 21 normalized landmarks, classifies the sign with a neural network, and shows the result on screen with optional text-to-speech. No gloves, depth sensors, or special hardware are needed.

> **Project status:** Review I complete. The static alphabet pipeline (data, model, live demo) is finished. Conversational word-level signs (LSTM on landmark sequences) are planned and not covered in this version.

---

## Features

- Real-time hand detection and landmark tracking (MediaPipe Hands, 21 keypoints)
- Landmark normalization, so predictions don't depend on hand position or distance from the camera
- Dense neural network classifier for 28 classes: A-Z, `del`, `space`
- Temporal smoothing (majority vote over a rolling 15-frame buffer) to prevent flicker
- On-screen prediction with confidence score, plus text-to-speech on keypress
- Webcam only, runs on a normal laptop CPU

---

## Results

| Metric | Value |
|---|---|
| Usable samples (after landmark extraction) | 66,270 |
| Classes | 28 (A-Z, `del`, `space`) |
| Train / Val / Test split | 46,389 / 9,940 / 9,941 (70/15/15, stratified) |
| Test accuracy | **98.98%** |
| Weighted precision / recall / F1 | 0.990 / 0.990 / 0.990 |
| Macro avg F1 | 0.989 |

Weakest classes on the test set: **N** (recall 0.922) and **M** (precision 0.950). The main confusion is N predicted as M. See [Known Limitations](#known-limitations).

The confusion matrix is saved to `models/confusion_matrix.png`.

---

## How It Works

```
Webcam frame
    -> MediaPipe Hands (21 landmarks x 3 coords = 63 features)
    -> Normalize (wrist-relative, scaled to unit size)
    -> Dense classifier (softmax over 28 classes)
    -> Confidence threshold + 15-frame majority vote
    -> On-screen label (+ optional text-to-speech)
```

**Model architecture:** `Input(63) -> Dense(128, ReLU) -> Dropout(0.3) -> Dense(64, ReLU) -> Dropout(0.2) -> Dense(28, softmax)`, trained with Adam and sparse categorical cross-entropy. Early stopping monitors validation accuracy (patience 8).

**Normalization:** each hand's landmarks are shifted so the wrist is the origin, then divided by the largest landmark distance. This makes features invariant to translation and scale.

---

## Project Structure

```
asl_interpreter/
├── asl_env/                    # virtual environment (not committed)
├── data/
│   ├── raw/asl_alphabet/       # downloaded Kaggle dataset
│   ├── landmarks/alphabet.csv  # extracted landmarks (63 features + label)
│   └── processed/              # alphabet_split.npz (train/val/test)
├── models/
│   ├── best_model.keras
│   ├── final_model.keras
│   ├── label_encoder.pkl
│   └── confusion_matrix.png
├── src/
│   ├── prepare_dataset.py      # normalize + stratified split
│   ├── train_model.py          # build and train the classifier
│   ├── evaluate.py             # classification report + confusion matrix
│   └── realtime_app.py         # live webcam demo
├── download_dataset.py         # Kaggle download
├── inspect_data.py             # sample images per class
├── extract_landmarks.py        # MediaPipe landmark extraction
├── check_detection_rate.py     # per-class landmark counts
├── test_setup.py               # environment sanity check
├── test_camera.py              # webcam sanity check
└── requirements.txt
```

---

## Setup

### Prerequisites
- Python 3.11
- A working webcam
- A free [Kaggle](https://www.kaggle.com) account (for the dataset download)

### Install

```powershell
python -m venv asl_env
.\asl_env\Scripts\Activate.ps1        # Linux/macOS: source asl_env/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### Pinned dependencies

```
numpy==1.26.4
opencv-contrib-python==4.10.0.84
mediapipe==0.10.21
tensorflow==2.17.1
scikit-learn==1.5.2
pandas==2.2.3
matplotlib==3.9.2
seaborn==0.13.2
pyttsx3==2.98
streamlit==1.39.0
kagglehub>=1.0.2
protobuf==4.25.5
```

**Why these versions:** `mediapipe==0.10.21` is used because newer MediaPipe releases removed the legacy `mp.solutions.hands` API this project relies on. `tensorflow==2.17.1` and `protobuf==4.25.5` are the combination compatible with it. `kagglehub>=1.0.2` is recommended over older versions because it supports resuming interrupted downloads (the dataset is about 1 GB).

### Verify the environment

```powershell
python test_setup.py      # should print "MediaPipe Hands OK" and library versions
python test_camera.py     # should print "Camera OK"
```

TensorFlow's oneDNN and MediaPipe's "Feedback manager" log lines are informational and safe to ignore.

---

## Usage: Full Pipeline

Run these in order from the project root.

### 1. Configure Kaggle credentials
Create a legacy API key at Kaggle -> Settings -> API and save `kaggle.json` to `C:\Users\<you>\.kaggle\` (or any folder, then point `KAGGLE_CONFIG_DIR` at it). Never commit or share this file or any API token.

### 2. Download the dataset
```powershell
python download_dataset.py
```
Uses the [ASL Alphabet dataset](https://www.kaggle.com/datasets/grassknoted/asl-alphabet) (about 87,000 images, 29 classes). If it times out, run the same command again; it resumes where it stopped. The class folders end up at `data/raw/asl_alphabet/asl_alphabet_train/asl_alphabet_train/` because the dataset is double-nested.

### 3. Inspect the data (optional)
```powershell
python inspect_data.py
```

### 4. Extract landmarks
```powershell
python extract_landmarks.py
```
Runs MediaPipe on every image (`min_detection_confidence=0.3`) and writes `data/landmarks/alphabet.csv`. Takes roughly 15-30 minutes. Disable laptop sleep during the run.

```powershell
python check_detection_rate.py
```
Shows how many usable samples each class produced.

### 5. Prepare and split the data
```powershell
python src/prepare_dataset.py
```
Drops the `nothing` class (no hand, so no landmarks), normalizes landmarks, and saves a stratified 70/15/15 split plus the label encoder.

### 6. Train
```powershell
python src/train_model.py
```
Saves `models/best_model.keras` and `models/final_model.keras`.

### 7. Evaluate
```powershell
python src/evaluate.py
```
Prints a per-class classification report and saves the confusion matrix.

### 8. Run the live demo
```powershell
python src/realtime_app.py
```

| Key | Action |
|---|---|
| `s` | Speak the currently detected letter |
| `q` | Quit |

Tips for best results: keep your hand well lit, fairly close to the camera, and hold each sign steady for about a second.

---

## Design Notes

- **Landmarks instead of raw pixels.** 63 numbers per sample trains fast on a CPU and is largely unaffected by background clutter, lighting, and skin tone.
- **Detect before flipping.** The mirrored view is only for display. Landmarks must be extracted from the original, unflipped frame, because the training images were not mirrored.
- **Low per-frame threshold plus voting.** Visually similar signs (M/N) rarely reach very high softmax confidence, so the live app uses `CONF_THRESHOLD = 0.40` and relies on the 15-frame majority vote (60% agreement required) for stability. If other letters start flickering, raise the threshold to about 0.55.
- **`nothing` handled implicitly.** When MediaPipe finds no hand, the app simply shows "No sign detected" and clears its buffer.

---

## Known Limitations

- **M, N, and `del` confusion.** These signs share a closed-fist handshape, differing mainly in how many fingers cover the thumb. That is hard to resolve from one 2D camera angle. MediaPipe also detects these hands less reliably (in extraction, M and N yielded fewer landmark samples than other letters). On held-out test data the model still scores 0.92-0.98 on these classes, but in live use M vs N confidence is often only around 45-55%, and the result is sensitive to hand angle and pose. Angling the hand slightly toward the camera and making the thumb position deliberate helps.
- **Static signs only.** J and Z involve motion in standard ASL; here they are classified from a single frame as the dataset provides.
- **Single signer in the live demo, single hand.** The model was trained on one dataset's signer(s); accuracy on new signers and lighting may be lower than the test-set figure.
- **Domain gap.** The test accuracy is measured on frames from the same dataset as training. Live webcam performance will differ.
- **RGB camera only.** Depth or multi-angle capture would help with M/N but is outside the project's scope.

---

## Roadmap

- [ ] Record a conversational word dataset (e.g., hello, thank_you, yes, no, please, sorry, help, good, bye, name)
- [ ] Train an LSTM on 30-frame landmark sequences for dynamic signs
- [ ] Combine alphabet and word recognition in one live app
- [ ] Streamlit GUI
- [ ] Per-signer evaluation and more diverse test data

---

## Tech Stack

Python 3.11, OpenCV, MediaPipe Hands, TensorFlow/Keras, scikit-learn, pandas, NumPy, Matplotlib, Seaborn, pyttsx3, KaggleHub.

## Dataset and Credits

- ASL Alphabet dataset on Kaggle: https://www.kaggle.com/datasets/grassknoted/asl-alphabet
- MediaPipe Hands by Google
- Related work: Pigou et al. (2015); Garcia and Viesca (2016); Kang et al. (2015); Rastgoo et al. (2021)
