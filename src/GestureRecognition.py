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
    
    def recognize(self, frame):
        mpImage = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
        timestamp = math.floor(time.time() * 1000)
        recognitionResult = self.recognizer.recognize_for_video(mpImage, timestamp)
        return self.drawLandmarks(frame, recognitionResult)


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
            
            annotatedImage = self.recognize(frame)
            cv2.imshow("webcam", annotatedImage)

            # Press 'q' to close webcam window.
            if cv2.waitKey(1) == ord('q'):
                print("Closing webcam.")
                break

        self.cap.release()
        cv2.destroyAllWindows()