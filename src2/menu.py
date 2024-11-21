from src2.gesture_recognition import GestureRecognition

import cv2
import numpy as np
import os

MUSIC_DIR = "music"
ALBUM_COVER_DIR = "album_cover"
COVER_SIZE = 300
GAP_SIZE = 25
HOVER_THRESH = 20

POINTER = cv2.imread('images/pointer.png', cv2.IMREAD_UNCHANGED)
POINTER_ALPHA = POINTER[:, :, 3]
POINTER = POINTER[:, :, :3]
POINTER_SIZE = (POINTER.shape[1], POINTER.shape[0])

RADIUS = 30

class Menu():
    def __init__(self, webcam, gestureRecognition):
        self.screenId = "select_music"
        self.webcam = webcam
        self.gestureRecognition = gestureRecognition
        self.musicList = sorted(os.listdir(MUSIC_DIR))
        self.coverList = sorted(os.listdir(ALBUM_COVER_DIR))
        self.hoverCount = {"posId": None, "count": 0}
    
    def selectMusic(self, frame):
        darkenedFrame = np.clip(frame * 0.25, 0, 255).astype(np.uint8)
        darkenedFrame = cv2.flip(darkenedFrame, 1)
        cv2.putText(darkenedFrame, "Select", (int(self.webcam.frameSize[0] / 10), int(self.webcam.frameSize[1] / 5)), cv2.FONT_HERSHEY_SIMPLEX, 5, (255, 255, 255), 10)
        cv2.putText(darkenedFrame, "Music", (int(self.webcam.frameSize[0] / 10), int(self.webcam.frameSize[1] / 5 * 2)), cv2.FONT_HERSHEY_SIMPLEX, 5, (255, 255, 255), 10)
        cv2.putText(darkenedFrame, "by", (int(self.webcam.frameSize[0] / 10), int(self.webcam.frameSize[1] / 5 * 3)), cv2.FONT_HERSHEY_SIMPLEX, 5, (255, 255, 255), 10)
        cv2.putText(darkenedFrame, "Pointing", (int(self.webcam.frameSize[0] / 10), int(self.webcam.frameSize[1] / 5 * 4)), cv2.FONT_HERSHEY_SIMPLEX, 5, (255, 255, 255), 10)

        x_offset = (darkenedFrame.shape[1] - COVER_SIZE) - GAP_SIZE
        y_offset = 0

        for coverFile in self.coverList:
            coverImage = cv2.imread(os.path.join(ALBUM_COVER_DIR, coverFile))
            coverImage = cv2.resize(coverImage, (COVER_SIZE, COVER_SIZE))
            y_offset += GAP_SIZE
            darkenedFrame[y_offset:y_offset + coverImage.shape[0], x_offset:x_offset + coverImage.shape[1]] = coverImage
            y_offset += coverImage.shape[0]

        resultFrame = self.drawPointer(darkenedFrame)
        resultFrame = self.drawCircularSector(darkenedFrame)

        return resultFrame
    
    def drawPointer(self, frame):
        hand = self.gestureRecognition.rightHand
        if hand["gesture"] == "Pointing_Up":
            x = int(min(hand["tipPosScaled"]["x"] * self.webcam.frameSize[0], self.webcam.frameSize[0] - POINTER_SIZE[0]))
            y = int(min(hand["tipPosScaled"]["y"] * self.webcam.frameSize[1], self.webcam.frameSize[1] - POINTER_SIZE[1]))
            x = max(0, x)
            y = max(0, y)

            for c in range(0, 3):  # Loop through BGR channels
                frame[y:y + POINTER_SIZE[1], x:x + POINTER_SIZE[0], c] = (POINTER_ALPHA / 255.0) * POINTER[:, :, c] + (1 - POINTER_ALPHA / 255.0) * frame[y:y + POINTER_SIZE[1], x:x + POINTER_SIZE[0], c]
        
        return frame

    def drawCircularSector(self, frame):
        if self.hoverCount["count"] > 0:
            hand = self.gestureRecognition.rightHand
            x = int(hand["tipPosScaled"]["x"] * self.webcam.frameSize[0])
            y = int(hand["tipPosScaled"]["y"] * self.webcam.frameSize[1])
            endAngle = int(360 * self.hoverCount["count"] / HOVER_THRESH)
            cv2.ellipse(frame, (x, y), (RADIUS, RADIUS), 0, 0, endAngle, (255, 255, 255), 15)
        
        return frame

    def checkHover(self):
        hand = self.gestureRecognition.rightHand
        if hand["gesture"] == "Pointing_Up":
            tipPosFrame = (int(hand["tipPosScaled"]["x"] * self.webcam.frameSize[0]), int(hand["tipPosScaled"]["y"] * self.webcam.frameSize[1]))
            if tipPosFrame[0] > self.webcam.frameSize[0] - COVER_SIZE - GAP_SIZE:
                if tipPosFrame[1] < COVER_SIZE + GAP_SIZE:
                    if self.hoverCount["posId"] == 0:
                        self.hoverCount["count"] += 1
                    else:
                        self.hoverCount["posId"] = 0
                        self.hoverCount["count"] = 0
                elif COVER_SIZE + GAP_SIZE*2 < tipPosFrame[1] and tipPosFrame[1] < COVER_SIZE*2 + GAP_SIZE*2:
                    if self.hoverCount["posId"] == 1:
                        self.hoverCount["count"] += 1
                    else:
                        self.hoverCount["posId"] = 1
                        self.hoverCount["count"] = 0
                elif COVER_SIZE*2 + GAP_SIZE*3 < tipPosFrame[1] and tipPosFrame[1] < COVER_SIZE*3 + GAP_SIZE*4:
                    if self.hoverCount["posId"] == 2:
                        self.hoverCount["count"] += 1
                    else:
                        self.hoverCount["posId"] = 2
                        self.hoverCount["count"] = 0
            else:
                self.hoverCount["posId"] = None
                self.hoverCount["count"] = 0
        else:
            self.hoverCount["posId"] = None
            self.hoverCount["count"] = 0


        if self.hoverCount["count"] > HOVER_THRESH:
            musicFile = self.musicList[self.hoverCount["posId"]]
            selectedMusic = os.path.join(MUSIC_DIR, musicFile)
        else:
            selectedMusic = None
        
        return selectedMusic