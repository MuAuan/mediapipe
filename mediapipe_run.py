import mediapipe as mp
from PIL import Image
import cv2
import csv
import numpy as np
import torch

from mediapipe_mlp_last import LitHands

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

hands = mp_hands.Hands(
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5,
)

model = LitHands()
PATH = "hands_mlp.ckpt"
pretrained_model = model.load_from_checkpoint(PATH)
print(pretrained_model)
pretrained_model.freeze()
pretrained_model.eval()

image0 = cv2.imread('blank.jpg') #白紙を台紙にします

idx = 0
cap = cv2.VideoCapture(0)
while cap.isOpened():
    success, image = cap.read()
    image_blank = image0.copy() #白紙を台紙にします
    cv2.imwrite('./image/x/image_o' + str(idx) + '.png', image)
    if not success:
        print("Ignoring empty camera frame.")
        # If loading a video, use 'break' instead of 'continue'.
        continue

    # Flip the image horizontally for a later selfie-view display, and convert
    # the BGR image to RGB.
    image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
    # To improve performance, optionally mark the image as not writeable to
    # pass by reference.
    image.flags.writeable = False
    results = hands.process(image)

    # Draw the hand annotations on the image.
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    image_width, image_height = image.shape[1], image.shape[0]
    with open('./hands/sample_hands_results.csv',  'a', newline='') as f:
        landmark_point = []
        writer = csv.writer(f)
        if results.multi_hand_landmarks:
            idx += 1
            #print('Handedness:', results.multi_handedness)
            for hand_landmarks in results.multi_hand_landmarks:
                for index, landmark in enumerate(hand_landmarks.landmark):
                    landmark_x = min(int(landmark.x * image_width), image_width - 1)
                    landmark_y = min(int(landmark.y * image_height), image_height - 1)

                    landmark_ = landmark_x,landmark_y
                    landmark_point.append(landmark_x)
                    landmark_point.append(landmark_y)
                                               
                #print(landmark_point)
                a = np.array(landmark_point).astype(int)
                a = torch.from_numpy(a).float()
                #print(a.reshape(1,21,2))
                a = a[:42]
                results_ = pretrained_model(a[:].reshape(1,21,2))
                print(results_)
                preds = torch.argmax(results_)
                print(preds)
                landmark_point.append(preds)
                writer.writerow(np.array(landmark_point))
                
            mp_drawing.draw_landmarks(image_blank, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                                    
            cv2.imshow('MediaPipe Hands_{}'.format(preds), image_blank)
            cv2.imwrite('./'+'image/{}'.format(preds) +'/image{}_'.format(preds) + str(idx) + '.png', cv2.flip(image_blank, 1))

    if cv2.waitKey(5) & 0xFF == 27:
        break
hands.close()
cap.release()