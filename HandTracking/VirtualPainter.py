import numpy as np
import cv2
import os
import time

import HandTrackingModule as htm

# Initialize Webcam
cap = cv2.VideoCapture(0)
cap.set(3, 1300)  # Width
cap.set(4, 800)   # Height

# Previous Time (For FPS Calculation)
pTime = 0

# Initialize Hand Detector
detector = htm.handDetector(detectionCon=0.85)

# Load Paint Images
folderPath = "PaintImages"
myList = os.listdir(folderPath)
overlayList = [cv2.imread(f'{folderPath}/{imPath}') for imPath in myList]

# Initial Header (Toolbar Image)
header = overlayList[0]

# Color and Brush Settings
drawColor = (241, 155, 25)  # Default color
brushThickness = 15
eraserThickness = 50

# Previous Points
xp, yp = 0, 0

# Initialize Canvas
_, img = cap.read()  # Read a frame to get the correct shape
imgCanvas = np.zeros_like(img)

while True:
    # 1. Capture Frame
    success, img = cap.read()
    img = cv2.flip(img, 1)  # Flip for better usability

    # 2. Find Hand
    img = detector.findHands(img)
    lmList = detector.findPosition(img, draw=False)

    if len(lmList) != 0:
        # Get Tip Coordinates
        x1, y1 = lmList[8][1:]   # Index finger tip
        x2, y2 = lmList[12][1:]  # Middle finger tip

        # 3. Detect Which Fingers Are Up
        fingers = detector.fingersUp()

        if len(fingers) >= 3:  # Ensure list is long enough
            # Selection Mode (Two Fingers Up)
            if fingers[1] and fingers[2]:
                xp, yp = 0, 0  # Reset drawing points

                # Check for Click in Toolbar
                if y1 < 140:
                    if 110 < x1 < 334:
                        header = overlayList[0]
                        drawColor = (241, 155, 25)
                    elif 430 < x1 < 658:
                        header = overlayList[1]
                        drawColor = (255, 0, 0)  
                    elif 770 < x1 < 960:
                        header = overlayList[2]
                        drawColor = (214, 0, 211)
                    elif 1070 < x1 < 1230:
                        header = overlayList[3]
                        drawColor = (255, 255, 255)  # White

                cv2.rectangle(img, (x1, y1 - 25), (x2, y2 + 25), drawColor, cv2.FILLED)

            # Drawing Mode (Only Index Finger Up)
            if fingers[1] and not fingers[2]:
                cv2.circle(img, (x1, y1), 15, drawColor, cv2.FILLED)

                if xp == 0 and yp == 0:
                    xp, yp = x1, y1

                # Erasing or Drawing
                if drawColor == (255, 255, 255):  # Eraser
                    cv2.line(img, (xp, yp), (x1, y1), drawColor, eraserThickness)
                    cv2.line(imgCanvas, (xp, yp), (x1, y1), (0, 0, 0), eraserThickness)
                else:
                    cv2.line(img, (xp, yp), (x1, y1), drawColor, brushThickness)
                    cv2.line(imgCanvas, (xp, yp), (x1, y1), drawColor, brushThickness)

                xp, yp = x1, y1  # Update previous position

    # 4. Process Canvas
    imgGray = cv2.cvtColor(imgCanvas, cv2.COLOR_BGR2GRAY)
    _, imgInv = cv2.threshold(imgGray, 50, 255, cv2.THRESH_BINARY_INV)
    imgInv = cv2.cvtColor(imgInv, cv2.COLOR_GRAY2BGR)

    # Ensure Matching Sizes
    imgCanvas = cv2.resize(imgCanvas, (img.shape[1], img.shape[0]))
    imgInv = cv2.resize(imgInv, (img.shape[1], img.shape[0]))

    img = cv2.bitwise_and(img, imgInv)
    img = cv2.bitwise_or(img, imgCanvas)

    # 5. Display Toolbar/Header
    frame_width = img.shape[1]
    header_resized = cv2.resize(header, (frame_width, 144))
    img[0:144, 0:frame_width] = header_resized

    # 6. Show Images
    cv2.imshow("Virtual Painter", img)
    # cv2.imshow("Canvas", imgCanvas)  # Debugging

    cv2.waitKey(1)
