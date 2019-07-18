import cv2
import os

if not os.path.exists('images/'):
    os.mkdir('images/')
video_capture = cv2.VideoCapture('videos/0701/3_1.MP4')
cnt = 0
while True:
    ret, f = video_capture.read()
    if ret != True:
        break
    cv2.imwrite('images/' + str(cnt) + '.jpg', f)
    cnt += 1
    print(cnt)