import mediapipe as mp
import cv2
import mediapipe as mp
mp_drawing = mp.solutions.drawing_utils

mp_face_mesh = mp.solutions.face_mesh
# For webcam input:
face_mesh = mp_face_mesh.FaceMesh(
    min_detection_confidence=0.5, min_tracking_confidence=0.5)
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)    

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(
    min_detection_confidence=0.5, min_tracking_confidence=0.5)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5,
)

image_blank = cv2.imread('blank.jpg')
sk = 0
cap = cv2.VideoCapture(0)
while cap.isOpened():
    success, image = cap.read()
    image_blank = cv2.imread('blank.jpg')

    if not success:
        print("Ignoring empty camera frame.")
        # If loading a video, use 'break' instead of 'continue'.
        continue
    #cv2.imshow('Camera', image)
    image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
    cv2.imshow('Camera', image)
    image.flags.writeable = False
    results_face = face_mesh.process(image)
    results_pose = pose.process(image)
    results_hands = hands.process(image)
    
    # Draw the face mesh annotations on the image.
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    if results_face.multi_face_landmarks:
        for face_landmarks in results_face.multi_face_landmarks:
            mp_drawing.draw_landmarks(
                image=image_blank,
                landmark_list=face_landmarks,
                connections=mp_face_mesh.FACE_CONNECTIONS,
                landmark_drawing_spec=drawing_spec,
                connection_drawing_spec=drawing_spec)
        cv2.imshow('MediaPipe FaceMesh', image_blank)
        cv2.imwrite('./image/blank/face/image'+ str(sk) + '.png', image_blank)
    image_width, image_height = image.shape[1], image.shape[0]
    if results_hands.multi_hand_landmarks:
        for hand_landmarks in results_hands.multi_hand_landmarks:
            #print('Handedness:', results.multi_handedness)
            mp_drawing.draw_landmarks(image_blank, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            cv2.imshow('MediaPipe Hands', image_blank)
            cv2.imwrite('./image/blank/facehands/image'+ str(sk) + '.png', image_blank)
    mp_drawing.draw_landmarks(
        image_blank, results_pose.pose_landmarks, mp_pose.POSE_CONNECTIONS)
    cv2.imshow('MediaPipe Pose', image_blank)        
    sk += 1   
    #cv2.imshow('amera_mirror', image)
    if cv2.waitKey(5) & 0xFF == 27:
        break
face_mesh.close()
hands.close()
pose.close()
cap.release()
