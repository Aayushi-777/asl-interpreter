import cv2
cap=cv2.VideoCapture(0)
ret, frame=cap.read()
print("Camera Ok" if ret else "Camera NOT detected")
cap.release()