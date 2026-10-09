# test_setup.py
import mediapipe as mp, cv2, tensorflow as tf

hands = mp.solutions.hands.Hands(static_image_mode=True, max_num_hands=2)
print("MediaPipe Hands OK")
print("OpenCV:", cv2.__version__, "| TensorFlow:", tf.__version__)