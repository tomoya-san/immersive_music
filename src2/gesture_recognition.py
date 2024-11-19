import math
import time
import cv2
import mediapipe as mp

from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import (
    GestureRecognizer,
    GestureRecognizerOptions,
    GestureRecognizerResult,
    RunningMode,
)

MODEL_PATH = "models/gesture_recognizer.task"

class GestureRecognition():
    def __init__(self):
        self.options = None
        self.recognizer = None
        self.rightHand = {"posScaled": None, "gesture": None, "openness": None}
        self.leftHand = {"posScaled": None, "gesture": None, "openness": None}

    def initialize(self):
        self.options = GestureRecognizerOptions(
            base_options = BaseOptions(model_asset_path=MODEL_PATH),
            running_mode = RunningMode.VIDEO,
            num_hands = 2,
            min_hand_detection_confidence = 0.5,
            min_hand_presence_confidence = 0.5,
            min_tracking_confidence = 0.5,
        )
        self.recognizer = GestureRecognizer.create_from_options(self.options)

    def recognize(self, frame):
        mpImage = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
        timestamp = math.floor(time.time() * 1000)
        result = self.recognizer.recognize_for_video(mpImage, timestamp)
        
        return result

    def saveResult(self, result):
        gestureList = result.gestures
        landmarksList = result.hand_landmarks
        handednessList = result.handedness

        for gesture, landmarks, handedness in zip(gestureList, landmarksList, handednessList):
            # right or left hand
            handedness = handedness[0].display_name

            # scaled position of the hand 0.0 ~ 1.0 for x and y
            xScaled = 1 - landmarks[9].x
            yScaled = landmarks[9].y
            zScaled = landmarks[9].z
            posScaled = {"x": xScaled, "y": yScaled, "z": zScaled}

            # how open the hand is (0 -> closed, 1 -> opened)
            wrist = (landmarks[0].x, landmarks[0].y, landmarks[0].z)
            middleMcp = (landmarks[9].x, landmarks[9].y, landmarks[9].z)
            handSize = 1.38 * math.dist(wrist, middleMcp)
            thumbTip = (landmarks[4].x, landmarks[4].y, landmarks[4].z)
            pinkyTip = (landmarks[20].x, landmarks[20].y, landmarks[20].z)
            handOpenness = math.dist(thumbTip, pinkyTip) / handSize

            if handedness == "Right":
                self.rightHand["posScaled"] = posScaled
                self.rightHand["gesture"] = gesture[0].category_name
                self.rightHand["openness"] = handOpenness

            
            elif handedness == "Left":
                self.leftHand["posScaled"] = posScaled
                self.leftHand["gesture"] = gesture[0].category_name
                self.leftHand["openness"] = handOpenness
