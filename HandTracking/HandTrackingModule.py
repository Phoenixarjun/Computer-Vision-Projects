import cv2 
import time
import mediapipe as mp
import math

class handDetector():
  def __init__(self,mode=False, maxHands=2, detectionCon=0.5, trackCon=0.5):
    self.mode = mode
    self.maxHands = maxHands
    self.detectionCon = detectionCon
    self.trackCon = trackCon
    

    self.mpHands = mp.solutions.hands
    self.hands = self.mpHands.Hands(
        static_image_mode=self.mode,
        max_num_hands=self.maxHands,
        min_detection_confidence=self.detectionCon,
        min_tracking_confidence=self.trackCon
    )

    self.mpDraw = mp.solutions.drawing_utils 
    self.tipIds = [4, 8, 12, 16, 20]


  def findHands(self,img, draw=True):
      imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
      self.results =  self.hands.process(imgRGB)

      if self.results.multi_hand_landmarks:
        for handLms in self.results.multi_hand_landmarks:
          if draw:
            self.mpDraw.draw_landmarks(img, handLms, 
                                      self.mpHands.HAND_CONNECTIONS)
      
      return img

  def findPosition(self, img, handNo=0, draw = True):

    self.lmList = []
    if self.results.multi_hand_landmarks:
      myHand = self.results.multi_hand_landmarks[handNo] 
      for id, lm in enumerate(myHand.landmark):
        h, w, c = img.shape
        cx, cy = int(lm.x*w), int(lm.y*h)
        self.lmList.append([id,cx,cy])

        if draw:
          cv2.circle(img,(cx,cy),5,(0,255,255),cv2.FILLED)
      
    return self.lmList
  
  
  def fingersUp(self):
    fingers = []
    
    if not self.lmList:
        return fingers  

    # Thumb
    if self.lmList[self.tipIds[0]][1] < self.lmList[self.tipIds[0]-1][1]:
        fingers.append(1)
    else:
        fingers.append(0)
    
    # 4 Fingers
    for id in range(1, 5):
        if self.lmList[self.tipIds[id]][2] < self.lmList[self.tipIds[id]-2][2]:
            fingers.append(1)
        else:
            fingers.append(0)

    return fingers


  def findDistance(self, p1, p2, img, draw=True,r=15, t=3):
    x1, y1 = self.lmList[p1][1:]
    x2, y2 = self.lmList[p2][1:]
    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

    if draw:
        cv2.line(img, (x1, y1), (x2, y2), (255, 0, 255), t)
        cv2.circle(img, (x1, y1), r, (255, 0, 255), cv2.FILLED)
        cv2.circle(img, (x2, y2), r, (255, 0, 255), cv2.FILLED)
        cv2.circle(img, (cx, cy), r, (0, 0, 255), cv2.FILLED)
        length = math.hypot(x2 - x1, y2 - y1)

    return length, img, [x1, y1, x2, y2, cx, cy]

  def overlayPNG(imgBack, imgFront, pos=[0, 0]):
      hf, wf, cf = imgFront.shape
      hb, wb, cb = imgBack.shape

      x1, y1 = max(pos[0], 0), max(pos[1], 0)
      x2, y2 = min(pos[0] + wf, wb), min(pos[1] + hf, hb)

      x1_overlay = 0 if pos[0] >= 0 else -pos[0]
      y1_overlay = 0 if pos[1] >= 0 else -pos[1]

      wf, hf = x2 - x1, y2 - y1

      if wf <= 0 or hf <= 0:
          return imgBack

      alpha = imgFront[y1_overlay:y1_overlay + hf, x1_overlay:x1_overlay + wf, 3] / 255.0
      inv_alpha = 1.0 - alpha

      imgRGB = imgFront[y1_overlay:y1_overlay + hf, x1_overlay:x1_overlay + wf, 0:3]

      for c in range(0, 3):
          imgBack[y1:y2, x1:x2, c] = imgBack[y1:y2, x1:x2, c] * inv_alpha + imgRGB[:, :, c] * alpha

      return imgBack

def main():
  pTime = 0 
  cTime = 0
  cap = cv2.VideoCapture(0)
  detector = handDetector()

  while True:
    success, img = cap.read()
    img = detector.findHands(img)
    lmList = detector.findPosition(img)
    
    if len(lmList) != 0:
      print(lmList[4])

    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime
    cv2.putText(img,str(int(fps)),(10,70), cv2.FONT_HERSHEY_SIMPLEX, 3, (255,0,0), 3)


    cv2.imshow("Image", img)
    cv2.waitKey(1)


if __name__ == "__main__":
  main()