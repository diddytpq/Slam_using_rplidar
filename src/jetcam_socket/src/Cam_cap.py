import sys
import cv2

cap = cv2.VideoCapture(2)

if not cap.isOpened():
    print('fail')

while True:
    _, frame = cap.read()
    cv2.imshow('test', frame)
    key = cv2.waitKey(1)
    if key == 27:
        break
