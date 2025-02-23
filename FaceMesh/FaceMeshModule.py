import cv2
import time
import mediapipe as mp


class FaceMeshDetector():
    
    def __init__(self, staticMode=False, maxFaces=2, minDetectionCon=0.5, minTrackCon=0.5):
        self.staticMode = staticMode
        self.maxFaces = maxFaces
        self.minDetectionCon = minDetectionCon
        self.minTrackCon = minTrackCon  

        self.mpDraw = mp.solutions.drawing_utils
        self.mpFaceMesh = mp.solutions.face_mesh
        self.faceMesh = self.mpFaceMesh.FaceMesh(
            static_image_mode=self.staticMode, 
            max_num_faces=self.maxFaces,
            min_detection_confidence=self.minDetectionCon,
            min_tracking_confidence=self.minTrackCon
        )
        self.drawSpec = self.mpDraw.DrawingSpec(thickness=1, circle_radius=2)

    def findFaceMesh(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.faceMesh.process(imgRGB)
        faces = []

        if self.results.multi_face_landmarks:
            for faceLms in self.results.multi_face_landmarks:
                face = []
                for id, lm in enumerate(faceLms.landmark):
                    h, w, c = img.shape 
                    x, y = int(lm.x * w), int(lm.y * h)
                    cv2.putText(img, str(id), (x, y), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 1)

                    face.append([x, y])

                faces.append(face)

                if draw:
                    self.mpDraw.draw_landmarks(img, faceLms, self.mpFaceMesh.FACEMESH_TESSELATION, 
                                              self.drawSpec, self.drawSpec)

        return img, faces  


def main():
    cap = cv2.VideoCapture(0)
    pTime = 0
    detector = FaceMeshDetector()

    while True:
        success, img = cap.read()
        if not success:
            break

        img, faces = detector.findFaceMesh(img)

        if faces:
            print(f"Detected {len(faces)} face(s)")

        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime

        cv2.putText(img, f'FPS: {int(fps)}', (20, 70), cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 2)

        cv2.imshow("Face Mesh", img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
