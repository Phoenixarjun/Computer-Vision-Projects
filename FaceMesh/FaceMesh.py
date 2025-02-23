import cv2
import time
import mediapipe as mp

cap = cv2.VideoCapture(0)
pTime = 0

mpDraw = mp.solutions.drawing_utils
mpFaceMesh = mp.solutions.face_mesh
faceMesh = mpFaceMesh.FaceMesh(max_num_faces=2)
drawSpec = mpDraw.DrawingSpec(thickness=1, circle_radius=2)


while True:
    success, img = cap.read()
    if not success:
        break

    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  
    results = faceMesh.process(imgRGB)

    if results.multi_face_landmarks:
        for faceLms in results.multi_face_landmarks:
            mpDraw.draw_landmarks(img, faceLms, mpFaceMesh.FACE_CONNECTIONS,
                                  drawSpec.drawSpec)  
            for id,lm in enumerate(faceLms.landmark):
                # print(lm) 
                w, h, c = img.shape
                x, y = (lm.x * w), (lm.y * h)
                print(id,x,y)

    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime

    cv2.putText(img, f'FPS: {int(fps)}', (20, 70), cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 2)

    cv2.imshow("img", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):  
        break

cap.release()
cv2.destroyAllWindows()
