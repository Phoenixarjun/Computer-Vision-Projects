import cv2 
import numpy as np
import time
import autopy
import HandTrackingModule as htm


wCam, hCam = 640, 480
frameR = 100 #Frame Reduction
smoothening = 6


pLocX, pLocY = 0, 0
cLocX, cLocY = 0, 0


cap = cv2.VideoCapture(0)
cap.set(3,wCam)
cap.set(4,hCam)
detector = htm.handDetector(maxHands=1)
wScr, hScr = autopy.screen.size()
# print(wScr, hScr)

pTime = 0

while True:
  # 1. Find the landmarks
  success, img = cap.read()
  img = detector.findHands(img)

  lmList = detector.findPosition(img)

  #2. Get the tip of the index and middle finger
  if len(lmList) != 0:
    x1, y1 = lmList[8][1:]
    x2, y2 = lmList[12][1:]

    #3. Check which fingers are up
    fingers = detector.fingersUp()
    cv2.rectangle(img, (frameR,frameR),(wCam - frameR, hCam - frameR),(255,0,255), 2)

    #4. Only Index Finger : Moving Mode
    if fingers[1] == 1 and fingers[2] == 0:

    
      #5. Convert Coordinates
      x3 = np.interp(x1,(frameR,wCam - frameR),(0,wScr))
      y3 = np.interp(y1,(frameR,hCam - frameR),(0,hScr))
      #6. Smoothen Values
      cLocX = pLocX + (x3 - pLocX) / smoothening
      cLocY = pLocY + (y3 - pLocY) / smoothening


      #7. Move Mouse
      autopy.mouse.move(wScr - cLocX, cLocY)
      cv2.circle(img,(x1,y1),15,(255,0,255),cv2.FILLED)
      pLocX, pLocY = cLocX, cLocY

    #8. Both Index and middle fingers are up: Clicking Mode
    if fingers[1] == 1 and fingers[2] == 1:

      #9. Find distance between fingers
      length, img, lineInfo = detector.findDistance(8, 12, img)
      # print(length) 

      #10. Click mouse if distance short
      if length < 30:
        cv2.circle(img,(lineInfo[4],lineInfo[5]),15,(0,255,0),cv2.FILLED)

        autopy.mouse.click()


  #Frame Rate
  cTime = time.time()
  fps = 1/(cTime - pTime)
  pTime = cTime

  cv2.imshow("Img",img)


  if cv2.waitKey(1) & 0xFF == ord('q'):
    break