import cv2
import numpy as np
import HandTrackingModule as htm
import time

class Button:
  def __init__(self, pos, width, height, value):
    self.pos = pos
    self.width = width
    self.height = height
    self.value = value


  def draw(self,img):
    cv2.rectangle(img, self.pos,(self.pos[0] + self.width, self.pos[1] + self.height),
                  (225,225,225),cv2.FILLED)
    cv2.rectangle(img, self.pos,(self.pos[0] + self.width, self.pos[1] + self.height),
                  (50,50,50),3)
    cv2.putText(img,self.value,(self.pos[0] + 35 ,self.pos[1] + 50),cv2.FONT_HERSHEY_PLAIN,
                3,(50,50,50),3)
    
  def checkClick(self, x, y):

    if self.pos[0] < x < self.pos[0] + self.width and \
      self.pos[1] < y < self.pos[1] + self.height:
      cv2.rectangle(img, self.pos,(self.pos[0] + self.width, self.pos[1] + self.height),
                    (255,255,255),cv2.FILLED)
      cv2.putText(img,self.value,(self.pos[0] + 30 ,self.pos[1] + 60),cv2.FONT_HERSHEY_PLAIN,
                  3,(0,0,0),5)
      return True
    else:
      return False



cap = cv2.VideoCapture(0)
cap.set(3,1280)
cap.set(4,720)
pTime = 0
detector = htm.handDetector(detectionCon=0.8, maxHands=1)


# Creating Buttons
buttonListValues = [['7','8','9','*'],
                    ['4','5','6','-'],
                    ['1','2','3','+'],
                    ['0','/','C','=']]
buttonList = []
for x in range(4):
  for y in range(4):
    xpos = x * 100 + 800
    ypos = y * 100 + 150
    buttonList.append(Button((xpos,ypos),100,100,buttonListValues[y][x]))

# Variables
myEquation = ''
delayCounter = 0

while True:
  success, img = cap.read()
  img = cv2.flip(img, 1)

  img = detector.findHands(img)

  # Draw all buttons
  cv2.rectangle(img, (800,50),(800 + 400, 70 + 100),
                  (225,225,225),cv2.FILLED)
  cv2.rectangle(img, (800,50),(800 + 400, 70 + 100),
                  (50,50,50),3)
  for btn in buttonList:
    btn.draw(img)

  #Check for hand
  lmList = detector.findPosition(img, draw=False)

  if len(lmList) != 0:
    length, img , _ = detector.findDistance(8,12,img)
    print(length)
    x,y = lmList[8][1:]
    if length < 50:
      for i, btn in enumerate(buttonList):
        if btn.checkClick(x,y) and delayCounter == 0:
          # We have 2D arr but we use 1D here so we need to calculate x and y
          currValue = buttonListValues[int(i%4)][int(i/4)]
          if currValue == '=':
            myEquation = str(eval(myEquation))
          elif currValue == 'C':
            myEquation = ''
          else:
            myEquation += currValue
          delayCounter = 1 

  # Avoid Duplicates
  if delayCounter != 0:
    delayCounter += 1
    if delayCounter > 10:
      delayCounter = 0

  #Display the Equation/Result
  cv2.putText(img,myEquation,(810, 120),cv2.FONT_HERSHEY_PLAIN,
                3,(50,50,50),3)



  cTime = time.time()
  fps = 1/(cTime - pTime)
  pTime = cTime

  cv2.imshow("img",img)

  if cv2.waitKey(1) and 0xFF == ord('q'):
    break

