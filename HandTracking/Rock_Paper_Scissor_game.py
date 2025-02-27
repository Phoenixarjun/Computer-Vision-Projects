import cv2
import HandTrackingModule as htm  
import time
import numpy as np
import cvzone

# Initialize webcam
cap = cv2.VideoCapture(0)
cap.set(3, 640)  # Width
cap.set(4, 480)  # Height


detector = htm.handDetector(maxHands=1)

# Game variables
timer = 0
stateResult = False  
startGame = False  
scores = [0, 0]  # [AI score, Player score]
draw = False  
while True:
    # Load background image
    imgBg = cv2.imread("RockPaperScissorImages/BG.png")  
    success, img = cap.read()

    # Resize and crop the webcam image
    imgScaled = cv2.resize(img, (0, 0), None, 0.875, 0.875)
    imgScaled = imgScaled[:, 80:480]  # Crop to match the overlay area

    img = detector.findHands(imgScaled)
    lmList = detector.findPosition(imgScaled, draw=False)

    if startGame:
        if stateResult is False:
            timer = time.time() - initialTime
            if not draw:
                cv2.putText(imgBg, str(int(timer)), (605, 435), cv2.FONT_HERSHEY_PLAIN, 6, (255, 0, 255), 4)

            # After 3 seconds, show the result
            if timer > 3:
                stateResult = True
                timer = 0

                # Determine player's move based on hand gesture
                if len(lmList) >= 3:
                    playerMove = None
                    fingers = detector.fingersUp()
                    print(fingers)
                    if fingers == [1, 0, 0, 0, 0]:  # Rock
                        playerMove = 1
                    elif fingers == [0, 1, 1, 1, 1]:  # Paper
                        playerMove = 2
                    elif fingers == [1, 1, 1, 0, 0]:  # Scissor
                        playerMove = 3

                    # AI's random move
                    randomNumber = np.random.randint(1, 4)
                    imgAI = cv2.imread(f'RockPaperScissorImages/{randomNumber}.png', cv2.IMREAD_UNCHANGED)

                    if playerMove is not None:
                        imgBg = cvzone.overlayPNG(imgBg, imgAI, (149, 310))

                    # Determine the winner
                    if (playerMove == 1 and randomNumber == 3) or \
                      (playerMove == 2 and randomNumber == 1) or \
                      (playerMove == 3 and randomNumber == 2):
                        scores[1] += 1  # Player wins
                        draw = False

                    elif (playerMove == 3 and randomNumber == 1) or \
                        (playerMove == 1 and randomNumber == 2) or \
                        (playerMove == 2 and randomNumber == 3):
                        scores[0] += 1  # AI wins
                        draw = False

                    else:
                        draw = True  # Draw


    imgBg[234:654, 795:1195] = imgScaled


    if stateResult:
        imgBg = cvzone.overlayPNG(imgBg, imgAI, (149, 310))

    # Display scores
    cv2.putText(imgBg, str(scores[0]), (410, 215), cv2.FONT_HERSHEY_PLAIN, 4, (255, 255, 255), 6)
    cv2.putText(imgBg, str(scores[1]), (1112, 215), cv2.FONT_HERSHEY_PLAIN, 4, (255, 255, 255), 6)

    if draw:
        cv2.putText(imgBg, 'Draw', (530, 435), cv2.FONT_HERSHEY_PLAIN, 6, (255, 0, 0), 4)
    else:
        cv2.putText(imgBg, str(int(timer)), (605, 435), cv2.FONT_HERSHEY_PLAIN, 6, (255, 0, 255), 4)


    cv2.imshow('Rock Paper Scissors', imgBg)


    key = cv2.waitKey(1)
    if key == ord('s'):  
        startGame = True
        initialTime = time.time()
        stateResult = False
        draw = False  
    elif key == ord('q'): 
        break

cap.release()
cv2.destroyAllWindows()