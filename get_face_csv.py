import mediapipe as mp
from PIL import Image
import cv2
import csv
import numpy as np

mp_drawing = mp.solutions.drawing_utils
mp_face_mesh = mp.solutions.face_mesh

# For webcam input:
face_mesh = mp_face_mesh.FaceMesh(
    min_detection_confidence=0.5, min_tracking_confidence=0.5)
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)  

idx = 0
cap = cv2.VideoCapture(0)
sk = 0
while cap.isOpened():
    """
    success, image = cap.read()
    if not success:
        print("Ignoring empty camera frame.")
        # If loading a video, use 'break' instead of 'continue'.
        continue
    """
    image = cv2.imread('./face/mayuyu/{}'.format(sk%10) + '.jpg')
    print(sk%10)
    sk += 1

    # Flip the image horizontally for a later selfie-view display, and convert
    # the BGR image to RGB.
    image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
    # To improve performance, optionally mark the image as not writeable to
    # pass by reference.
    image.flags.writeable = False
    #results = hands.process(image)
    results = face_mesh.process(image)

    # Draw the hand annotations on the image.
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    image_width, image_height = image.shape[1], image.shape[0]
    with open('./hands/sample_face1.csv',  'a', newline='') as f:
        landmark_point = []
        writer = csv.writer(f)
        if results.multi_face_landmarks:
            idx += 1
            print('Handedness:', results.multi_face_landmarks)
            for face_landmarks in results.multi_face_landmarks:
                mp_drawing.draw_landmarks(
                    image=image,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACE_CONNECTIONS,
                    landmark_drawing_spec=drawing_spec,
                    connection_drawing_spec=drawing_spec)
            
            
            for hand_landmarks in results.multi_face_landmarks:
                for index, landmark in enumerate(face_landmarks.landmark):
                    landmark_x = min(int(landmark.x * image_width), image_width - 1)
                    landmark_y = min(int(landmark.y * image_height), image_height - 1)

                    landmark_ = landmark_x,landmark_y #[idx,index, np.array((landmark_x, landmark_y))]
                    landmark_point.append(landmark_x)
                    landmark_point.append(landmark_y)

                print(landmark_point)
                writer.writerow(np.array(landmark_point))
            cv2.imshow('MediaPipe FaceMesh', image)
            cv2.imwrite('./image/face_image' + str(idx) + '.png', image)
    if cv2.waitKey(5) & 0xFF == 27:
        break
face_mesh.close()
cap.release()