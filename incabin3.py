import threading
#import concurrent.futures
#import pygame
import cv2
import dlib
import time, sys
import numpy as np
from keras.preprocessing.image import img_to_array
import imutils
from tensorflow.keras.models import load_model

import json
import tensorflow as tf
#pygame.mixer.init()
#songs = ['beeep.mp3', 'alarm2.mp3', 'welcomenote.mp3']

#pygame.mixer.music.load(songs[2])
#pygame.mixer.music.play(0)

face_downsample_ratio = 1.2
resize_height = 360
thresh = 0.43

model_path = "shape_predictor_68_face_landmarks.dat"
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(model_path)

left_eye_index = [36, 37, 38, 39, 40, 41]
right_eye_index = [42, 43, 44, 45, 46, 47]
mouth_index = [49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67]
chin_index = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17]

blink_count = 0
drowsy = 0
state = 0
blink_time = 0.02  # 200ms
drowsy_time = 0.07  # 900ms

drunk_count = 0
emotion_count=0


log_drowsy = {"timestamp_drowsy":[],"Drowsy":[]}
log_emotion = {"timestamp_emotion":[],"emotion":[]}
log_drunk = {"timestamp_drunk":[],"Drunkenness":[]}
detection_model_path = 'haarcascade_frontalface_default.xml'
emotion_model_path = 'models/_mini_XCEPTION.102-0.66.hdf5'
model1 = load_model('models/model2.hdf5')
face_detection = cv2.CascadeClassifier(cv2.data.haarcascades + detection_model_path)
emotion_classifier = load_model(emotion_model_path, compile=False)
emotions = ["angry", "disgust", "scared", "happy", "sad", "surprised", "neutral"]

def gstreamer_pipeline(capture_width=1280, capture_height=720, display_width=1280,
                       display_height=720, framerate=30, flip_method=0):
    return ('nvarguscamerasrc ! '
            'video/x-raw(memory:NVMM), '
            'width=(int)%d, height=(int)%d, '
            'format=(string)NV12, framerate=(fraction)%d/1 ! '
            'nvvidconv flip-method=%d ! '
            'video/x-raw, width=(int)%d, height=(int)%d, format=(string)BGRx ! '
            'videoconvert ! '
            'video/x-raw, format=(string)BGR ! appsink' % (
                capture_width, capture_height, framerate, flip_method, display_width, display_height))


def checkEyeStatus(landmarks):
    # Create a black image to be used as a mask for the eyes
    mask = np.zeros(frame.shape[:2], dtype=np.float32)

    # Create a convex hull using the points of the left and right eye
    hullLeftEye = []
    for i in range(0, len(left_eye_index)):
        hullLeftEye.append((landmarks[left_eye_index[i]][0], landmarks[left_eye_index[i]][1]))

    cv2.fillConvexPoly(mask, np.int32(hullLeftEye), 255)

    hullRightEye = []
    for i in range(0, len(right_eye_index)):
        hullRightEye.append((landmarks[right_eye_index[i]][0], landmarks[right_eye_index[i]][1]))

    cv2.fillConvexPoly(mask, np.int32(hullRightEye), 255);

    # find the distance between the tips of left eye
    lenLeftEyeX = landmarks[left_eye_index[3]][0] - landmarks[left_eye_index[0]][0];
    lenLeftEyeY = landmarks[left_eye_index[3]][1] - landmarks[left_eye_index[0]][1];

    lenLeftEyeSquare = lenLeftEyeX * lenLeftEyeX + lenLeftEyeY * lenLeftEyeY;

    # find the area under the eye region
    eyeRegionCount = cv2.countNonZero(mask)

    # normalize the area by the length of eye
    # The threshold will not work without the normalization
    # the same amount of eye opening will have more area if it is close to the camera
    normalizedCount = eyeRegionCount / np.float32(lenLeftEyeSquare)

    eyeStatus = 1  # 1 -> Open, 0 -> closed
    if (normalizedCount < thresh):
        eyeStatus = 0

    return eyeStatus


# simple finite state machine to keep track of the blinks. we can change the behaviour as needed.
def checkBlinkStatus(eyeStatus):
    global state, blink_count, drowsy

    # open state and false blink state
    if (state >= 0 and state <= falseBlinkLimit):
        # if eye is open then stay in this state
        if (eyeStatus):
            state = 0
        # else go to next state
        else:
            state += 1

    # closed state for (drowsyLimit - falseBlinkLimit) frames
    elif (state > falseBlinkLimit and state <= drowsyLimit):
        if (eyeStatus):
            state = 0
            blink_count += 1
        else:
            state += 1

    # Extended closed state -- drowsy
    else:
        if (eyeStatus):
            state = 0
            blink_count += 1
            drowsy = 0
        else:
            drowsy = 1


def getLandmarks(im):
    imSmall = cv2.resize(im, None,
                         fx=1.0 / face_downsample_ratio,
                         fy=1.0 / face_downsample_ratio,
                         interpolation=cv2.INTER_LINEAR)
    # detect faces
    rects = detector(imSmall, 0)
    if len(rects) == 0:
        return 1

    # scale the points before sending to the pose predictor as we will send the original image
    newRect = dlib.rectangle(int(rects[0].left() * face_downsample_ratio),
                             int(rects[0].top() * face_downsample_ratio),
                             int(rects[0].right() * face_downsample_ratio),
                             int(rects[0].bottom() * face_downsample_ratio))

    # Create an array for storing the facial points
    points = []
    [points.append((p.x, p.y)) for p in predictor(im, newRect).parts()]
    return points






#####################################################################################
# Calculate the FPS for initialization
# Different computers will have relatively different speeds
# Since all operations are on frame basis
# We want to find how many frames correspond to the blink and drowsy limit

# Reading some dummy frames to adjust the sensor to the lighting

spf=0.14830
print(spf)
print("Current SPF (seconds per frame) is {:.2f} ms".format(spf * 1000))

drowsyLimit = drowsy_time / spf
falseBlinkLimit = blink_time / spf
print("drowsyLimit {} ( {:.2f} ms) ,  False blink limit {} ( {:.2f} ms) ".format(drowsyLimit, drowsyLimit * spf * 1000,
                                                                                 falseBlinkLimit,
                                                                                (falseBlinkLimit + 1) * spf * 1000))
capture = cv2.VideoCapture('test.mp4')

#capture = cv2.VideoCapture(0)
count=0
#capture=cv2.VideoCapture(gstreamer_pipeline(flip_method=0),cv2.CAP_GSTREAMER)
while True:
    try: 
      ret, frame = capture.read()
      tfirst= time.time()
      print("start",tfirst)
      count+=1
      print("countcountcountcountcountcountcountcountcountcountcountcountcount",count)
      count_ratio=count%1
      if (count_ratio==0):
        print("inside count_ratiocount_ratiocount_ratiocount_ratiocount_ratiocount_ratiocount_ratiocount_ratio",count_ratio)
        drunk_count_ratio=0
        emotion_count_ratio=0
        #drunk_count+=1
        #emotion_count+=1
        #drunk_count_ratio=drunk_count%1
        #emotion_count_ratio=emotion_count%1
        #print("drunk_count_ratio",drunk_count_ratio)
        #print("emotion_count_ratio",emotion_count_ratio)
        height, width = frame.shape[:2]

        p1=time.time()


        image_resize = np.float32(height) / resize_height
        frame = cv2.resize(frame, None,
                           fx=1.0 / image_resize,
                           fy=1.0 / image_resize,
                           interpolation=cv2.INTER_LINEAR)
        # frame = cv2.flip(frame, -1)
        start = time.time()
        landmarks = getLandmarks(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)) #here...........

        # if face not detected
        if landmarks == 1:
            #cv2.putText(frame, "Unable to detect face, Please check proper lighting", (10, 50),
            #           cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 0, 0), 1, cv2.LINE_AA)
            #cv2.putText(frame, "Or Decrease FACE_DOWNSAMPLE_RATIO", (10, 150), cv2.FONT_HERSHEY_COMPLEX, 0.5,
            #            (255, 0, 0), 1, cv2.LINE_AA)
            cv2.imshow("Blink Detection Demo ", frame)
            if cv2.waitKey(1) & 0xFF == 27:
                break
            continue

        # check whether eye is open or close

        #eyeStatus = threading.Thread(target=checkEyeStatus,args=[landmarks])
        #BlinkStatus = threading.Thread(target=checkBlinkStatus,args=[eyeStatus])
        #eyeStatus.start()
        #BlinkStatus.start()

        eyeStatus = checkEyeStatus(landmarks) #here...................
        

        # pass the eyestatus to the state machine
        # to determine the blink count and drowsiness status
        
        checkBlinkStatus(eyeStatus) #here..................
        finish = time.time()
        print(f"Finished in {round(finish-start,2)} SECONDS")
        # Plot the eyepoints on the face for showing
        for i in range(0, len(left_eye_index)):
            cv2.circle(frame, (landmarks[left_eye_index[i]][0], landmarks[left_eye_index[i]][1]), 1, (0, 255, 0),
                       thickness=1, lineType=cv2.LINE_AA)
        for i in range(0, len(right_eye_index)):
            cv2.circle(frame, (landmarks[right_eye_index[i]][0], landmarks[right_eye_index[i]][1]), 1, (0, 255, 0),
                       thickness=1, lineType=cv2.LINE_AA)
        for i in range(0, len(mouth_index)):
            cv2.circle(frame, (landmarks[mouth_index[i]][0], landmarks[mouth_index[i]][1]), 1, (0, 255, 0),
                       thickness=1, lineType=cv2.LINE_AA)
        for i in range(0, len(chin_index)):
            cv2.circle(frame, (landmarks[chin_index[i]][0], landmarks[chin_index[i]][1]), 1, (255, 0, 0),
                       thickness=1, lineType=cv2.LINE_AA)
        print(" before if drowsy condition",time.time())
        
        if (drowsy):
            
            print("Drowsy!!!")
            cv2.putText(frame, "!!! DROWSY !!! ", (50, 100), cv2.FONT_HERSHEY_COMPLEX, 1.5, (0, 0, 255), 3,
                        cv2.LINE_AA)
            t1 = time.time()
            
            dictionary={"timestamp":t1,"Drowsy":"true"}
            with open("log.json", "a") as outfile:
                json.dump(dictionary,outfile)
                outfile.write('\n')


        



        else:
            cv2.putText(frame, "Blinks : {}".format(blink_count), (50, 50), cv2.FONT_HERSHEY_COMPLEX, .9,
                        (255, 0, 0),
                        2,
                        cv2.LINE_AA)
        #print(" after if drowsy condition",time.time())
        cv2.imshow("drowsy Detection Demo ", frame)
        #print('drowsy:',risk_factor)
        print("Blinks : {}".format(blink_count))
        if cv2.waitKey(1) & 0xFF == 27:
            break



        # hyper parameters for bounding box shape
        # loading models
        #print("b4 face_detection model loading and frame ",time.time())
        face_detection = cv2.CascadeClassifier(detection_model_path)
        #print("b4 face_detection model loading and frame ",time.time())
        frame1 = capture.read()[1]
        #frame1 = cv2.flip(frame1, -1)
        frame_width = int(capture.get(3))
        frame_height = int(capture.get(4))
        size = (frame_width, frame_height)

        # reading the frame
        frame1 = imutils.resize(frame1, width=600)
        gray = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        #print(" aftr face_detection model loading and frame ",time.time())
        #print("before face_detection",time.time())
        faces = face_detection.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30),
                                                flags=cv2.CASCADE_SCALE_IMAGE)
        #print("after face_detection",time.time())
        canvas = np.zeros((250, 300, 3), dtype="uint8")

        frame_width = int(capture.get(3))
        frame_height = int(capture.get(4))

        # Define the codec and create VideoWriter object.The output is stored in 'output.avi' file.
        frame_clone = frame1.copy()

        if len(faces) > 0:
            faces = sorted(faces, reverse=True, key=lambda x: (x[2] - x[0]) * (x[3] - x[1]))[0]
            (fX, fY, fW, fH) = faces
            # extract the roi of the face from the grayscale image,resize it to a fixed 28 x 28 pixels, and then prepare roi for classification by cnn

            roi = gray[fY:fY + fH, fX:fX + fW]
            roi = cv2.resize(roi, (64, 64))
            roi = roi.astype("float") / 255.0
            roi = img_to_array(roi)
            roi = np.expand_dims(roi, axis=0)
            
            
            if (drunk_count_ratio==0):
                
                prediction = model1.predict_classes(roi)


                if (prediction == [0]):
                    out = "drunk"
                    t1=time.time()
                
                    dictionary_drunk={"timestamp":t1,"Drunkenness":out}
                    with open("log.json", "a") as outfile:
                        json.dump(dictionary_drunk,outfile)
                        outfile.write('\n')
                elif (prediction == [1]):
                    out = "sober"
                    t1=time.time()
                
                    dictionary_sober={"timestamp":t1,"Drunkenness":out}
                    with open("log.json", "a") as outfile:
                        json.dump(dictionary_sober,outfile)
                        outfile.write('\n')
            if (emotion_count_ratio == 0):
                
                preds = emotion_classifier.predict(roi)[0]
                l1 = preds.argsort()[-1:][::-1]
                max_three_emo = []
                label = emotions[preds.argmax()]
                print("Okeeeeeey111111111111111.............")
                for (i, (emotion, prob)) in enumerate(zip(emotions, preds)):
                    # construct the label text
                    text = "{}: {:.2f}%".format(emotion, prob * 100)
                    w = int(prob * 300)
                    cv2.putText(frame, label, (fX + 10, fY), cv2.FONT_HERSHEY_COMPLEX, 0.45, (255, 255, 255), 1)
                    cv2.rectangle(frame, (fX, fY), (fX + fW, fY + fH), (255, 255, 255), 1)
                    cv2.rectangle(frame, (7, (i * 35) + 5), (w, (i * 35) + 35), (10, 10, 10), -1)
                    cv2.putText(frame, text, (10, (i * 35) + 23), cv2.FONT_HERSHEY_COMPLEX, 0.45, (255, 255, 255), 1)
                    cv2.putText(frame, out, (fX + 10, fY + 180), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 2) #commenting os out is inside if loop
                
                #t1 = time.time()
                #print(type(t1))
                #print(type(emotion_probability))
                #dictionary_emotion={"timestamp":t1,"emotion":label,"prob":emotion_probability.item()}
               # with open("log.json", "a") as outfile:
                #    json.dump(dictionary_emotion,outfile)
               #     outfile.write('\n')

                cv2.imshow('AI facial expression and drunkenness detection', frame)



        if cv2.waitKey(1) & 0xFF == 27:
            sys.exit()


    except Exception as e:
        print(e)

capture.release()

cv2.destroyAllWindows()


