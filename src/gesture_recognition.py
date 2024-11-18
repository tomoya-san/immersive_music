import math
import time
import numpy as np

import cv2
import mediapipe as mp

from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import (
    GestureRecognizer,
    GestureRecognizerOptions,
    GestureRecognizerResult,
    RunningMode,
)

from mediapipe.python.solutions import drawing_utils, drawing_styles, hands
from mediapipe.framework.formats import landmark_pb2

MODEL_PATH = "models/gesture_recognizer.task"

class GestureRecognition():
    def __init__(self):
        # Initialize Gesture Recognition
        self.options = GestureRecognizerOptions(
            base_options = BaseOptions(model_asset_path=MODEL_PATH),
            running_mode = RunningMode.VIDEO,
            num_hands = 2,
            min_hand_detection_confidence = 0.5,
            min_hand_presence_confidence = 0.5,
            min_tracking_confidence = 0.5,
            #result_callback = self.annotate,
        )
        self.recognizer = GestureRecognizer.create_from_options(self.options)

        # Initialize WebCam
        self.cap = cv2.VideoCapture(1)
        if (self.cap.isOpened()):
            print("Successfully opened webcam.")
        else:
            print("Error: Could not open webcam.")
            exit(0)

        self.frameSize = (self.cap.get(cv2.CAP_PROP_FRAME_WIDTH), self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.frameRate = int(self.cap.get(cv2.CAP_PROP_FPS))
        print(f"Frame Size: ({self.frameSize[0]}, {self.frameSize[1]})")
        print(f"Frame Rate: {self.frameRate}")

        self.rightHandPos = (0.75, 0.5)
        self.leftHandPos = (0.25, 0.5)
        self.rightGesture = "Unknown"
        self.leftGesture = "Unknown"
    
    def recognize(self, frame):
        mpImage = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
        timestamp = math.floor(time.time() * 1000)
        recognitionResult = self.recognizer.recognize_for_video(mpImage, timestamp)
        annotatedImage = self.getHandPosition(frame, recognitionResult)
        annotatedImage = self.drawBandPass(annotatedImage)
        self.getHandOpenness(recognitionResult)
        return annotatedImage
    
    def getHandOpenness(self, recognitionResult):
        landmarksList = recognitionResult.hand_landmarks
        handednessList = recognitionResult.handedness

        for index, landmarks in enumerate(landmarksList):
            handedness = handednessList[index][0].display_name

            wrist = (landmarks[0].x, landmarks[0].y, landmarks[0].z)
            middleMcp = (landmarks[9].x, landmarks[9].y, landmarks[9].z)
            handSize = 1.38 * math.dist(wrist, middleMcp)

            thumbTip = (landmarks[4].x, landmarks[4].y, landmarks[4].z)
            pinkyTip = (landmarks[20].x, landmarks[20].y, landmarks[20].z)
            
            handOpenness = math.dist(thumbTip, pinkyTip) / handSize
            if handedness == "Right":
                self.rightOpenness = handOpenness
            elif handedness == "Left":
                self.leftOpenness = handOpenness



    def getHandPosition(self, image, recognitionResult):
        gestureList = recognitionResult.gestures
        landmarksList = recognitionResult.hand_landmarks
        handednessList = recognitionResult.handedness
        annotatedImage = np.copy(image)
        annotatedImage = cv2.flip(annotatedImage, 1)

        for index, landmarks in enumerate(landmarksList):
            gesture = gestureList[index][0].category_name
            #print(gesture)
            handedness = handednessList[index][0].display_name

            xScaled = 1 - landmarks[9].x
            yScaled = landmarks[9].y
            posScaled = (xScaled, yScaled)

            xFrame = int(self.frameSize[0] * xScaled)
            yFrame = int(self.frameSize[1] * yScaled)
            posFrame = (xFrame, yFrame)

            cv2.circle(annotatedImage, center=posFrame, radius=10, color=(0,255,0), thickness=3)
            cv2.putText(annotatedImage, f"{handedness}", posFrame, cv2.FONT_HERSHEY_DUPLEX, 1.0, (0,255,0))
            #cv2.putText(annotatedImage, f"({coords[0]}, {coords[1]})", coords, cv2.FONT_HERSHEY_DUPLEX, 1.0, (0,255,0))
            
            if handedness == "Right":
                self.rightHandPos = posScaled
                self.rightGesture = gesture
            elif handedness == "Left":
                self.leftHandPos = posScaled
                self.leftGesture = gesture

        return annotatedImage

    def drawBandPass(self, image):
        topLeft = (int(self.leftHandPos[0] * self.frameSize[0]), int(self.frameSize[1] / 10))
        bottomRight = (int(self.rightHandPos[0] * self.frameSize[0]), int(self.frameSize[1] / 10 + 100))

        annotatedImage = cv2.rectangle(image, topLeft, bottomRight, (self.leftHandPos[0] * 255, 0, self.rightHandPos[0] * 255), -1)

        return annotatedImage
    
    def drawLandmarks(self, image, recognitionResult):
        landmarksList = recognitionResult.hand_landmarks
        handednessList = recognitionResult.handedness
        annotatedImage = np.copy(image)

        for landmarks in landmarksList:
            for point in landmarks:
                x = int(self.frameSize[0] * point.x)
                y = int(self.frameSize[1] * point.y)
                coords = (x, y)
                cv2.circle(annotatedImage, center=coords, radius=10, color=(255,0,0), thickness=3)

        return annotatedImage
    
    def startRecognition(self):
        while self.cap.isOpened():
            ret, frame = self.cap.read()

            if ret == False:
                print('Error: Could not receive frame.')
                break
            
            # frame = cv2.flip(frame, 1)
            annotatedImage = self.recognize(frame)
            cv2.imshow("webcam", annotatedImage)

            # Press 'q' to close webcam window.
            if cv2.waitKey(1) == ord('q'):
                print("Closing webcam.")
                break

        self.cap.release()
        cv2.destroyAllWindows()