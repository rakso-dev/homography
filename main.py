import cv2
import numpy as np
import time

video = cv2.VideoCapture("./video2.mp4")
prev_time = 0
nframe_time = 0
while(video.isOpened()):
    ret, frame = video.read()

    if not ret:
        break

    img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    img = cv2.medianBlur(img, 15)
    img = cv2.GaussianBlur(img, (11, 11), 0)
    img = cv2.Canny(img, 0, 20)
    img = cv2.dilate(img, None)
    contours, _ = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    max_contour = max(contours, key=cv2.contourArea)
    image = img.copy()
    #vertex = cv2.goodFeaturesToTrack(image, 4)
    image = cv2.drawContours(image, [max_contour], -1, 255, thickness=-1)

    components = cv2.connectedComponentsWithStats(image, 4, cv2.CV_32S)
    nobjects = components[0]
    labels = components[1]
    stats = components[2]
    mask = np.uint(255 * [np.argmax(stats[:4][:1]) + 1 == labels])


    nframe_time = time.time()
    fps = 1/(nframe_time - prev_time)
    print(int(fps))
    prev_time = nframe_time

    #cv2.imshow('Original', frame)
    cv2.imshow('Canny', img)
    cv2.imshow('contours', mask)
    #cv2.imshow('Gray', img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video.release()
cv2.destroyAllWindows()
