import cv2
import numpy as np
import os

# for music selection
MUSIC_DIR = "music"
ALBUM_COVER_DIR = "album_cover"
COVER_SIZE = 300
GAP_SIZE = 25

# for right pointing hand
POINTER = cv2.imread('images/pointer.png', cv2.IMREAD_UNCHANGED)
POINTER_ALPHA = POINTER[:, :, 3]
POINTER = POINTER[:, :, :3]
POINTER_SIZE = (POINTER.shape[1], POINTER.shape[0])

# for hovering
HOVER_THRESH = 20
RADIUS = 30

# for volume bar
VOLUME_BAR_POS = (100, 300)

# for pausing screen
TRIANGLE_SIZE = 200

# for showing detected hands
HAND_SIZE = (200, 200)
RIGHT_HAND = cv2.imread("images/right_hand_open.png", cv2.IMREAD_UNCHANGED)
LEFT_HAND = cv2.imread("images/left_hand_open.png", cv2.IMREAD_UNCHANGED)
RIGHT_HAND = cv2.resize(RIGHT_HAND, HAND_SIZE)
LEFT_HAND = cv2.resize(LEFT_HAND, HAND_SIZE)
RIGHT_HAND_ALPHA = RIGHT_HAND[:, :, 3]
RIGHT_HAND = RIGHT_HAND[:, :, :3]
LEFT_HAND_ALPHA = LEFT_HAND[:, :, 3]
LEFT_HAND = LEFT_HAND[:, :, :3]


class Menu():
    def __init__(self, webcam, gestureRecognition):
        self.screenId = "select_music"
        self.webcam = webcam
        self.gestureRecognition = gestureRecognition
        self.musicList = sorted(os.listdir(MUSIC_DIR))
        self.coverList = sorted(os.listdir(ALBUM_COVER_DIR))
        self.musicHoverCount = {"posId": None, "count": 0}
        self.pauseHoverCount = 0
        self.quitHoverCount = 0
    
    # returns a screen for the paused menu
    def pausingMusic(self, frame):
        # darken the frame
        darkenedFrame = np.clip(frame * 0.25, 0, 255).astype(np.uint8)
        darkenedFrame = cv2.flip(darkenedFrame, 1)

        # draw a restart icon
        width, height = int(self.webcam.frameSize[0]), int(self.webcam.frameSize[1])
        centerX, centerY = width // 2, height // 2
        points = np.array([
            [centerX - TRIANGLE_SIZE // 2, centerY - TRIANGLE_SIZE // 2],
            [centerX - TRIANGLE_SIZE // 2, centerY + TRIANGLE_SIZE // 2],
            [centerX + TRIANGLE_SIZE // 2, centerY],
        ])
        cv2.fillPoly(darkenedFrame, [points], (255, 255, 255)) 

        return darkenedFrame
    
    # returns a screen for the music selection menu
    def selectMusic(self, frame):
        # darken the frame
        darkenedFrame = np.clip(frame * 0.25, 0, 255).astype(np.uint8)
        darkenedFrame = cv2.flip(darkenedFrame, 1)

        # write text
        cv2.putText(darkenedFrame, "Select", (int(self.webcam.frameSize[0] / 10), int(self.webcam.frameSize[1] / 5)), cv2.FONT_HERSHEY_SIMPLEX, 5, (255, 255, 255), 10)
        cv2.putText(darkenedFrame, "Music", (int(self.webcam.frameSize[0] / 10), int(self.webcam.frameSize[1] / 5 * 2)), cv2.FONT_HERSHEY_SIMPLEX, 5, (255, 255, 255), 10)
        cv2.putText(darkenedFrame, "by", (int(self.webcam.frameSize[0] / 10), int(self.webcam.frameSize[1] / 5 * 3)), cv2.FONT_HERSHEY_SIMPLEX, 5, (255, 255, 255), 10)
        cv2.putText(darkenedFrame, "Pointing", (int(self.webcam.frameSize[0] / 10), int(self.webcam.frameSize[1] / 5 * 4)), cv2.FONT_HERSHEY_SIMPLEX, 5, (255, 255, 255), 10)

        # draw music album covers
        x_offset = (darkenedFrame.shape[1] - COVER_SIZE) - GAP_SIZE
        y_offset = 0

        for coverFile in self.coverList:
            coverImage = cv2.imread(os.path.join(ALBUM_COVER_DIR, coverFile))
            coverImage = cv2.resize(coverImage, (COVER_SIZE, COVER_SIZE))
            y_offset += GAP_SIZE
            darkenedFrame[y_offset:y_offset + coverImage.shape[0], x_offset:x_offset + coverImage.shape[1]] = coverImage
            y_offset += coverImage.shape[0]

        # draw pointing right hand
        resultFrame = self.drawPointer(darkenedFrame)

        # draw hovering animation
        resultFrame = self.drawCircularSector(darkenedFrame, self.musicHoverCount["count"])

        return resultFrame
    
    # darken the surrounding for band filter animation
    def darkenSurrounding(self, frame):
        hands = self.gestureRecognition
        if hands.rightHand["gesture"] != None and hands.leftHand["gesture"] != None:
            mask = np.zeros_like(frame, dtype=np.uint8)
            rightX = int(hands.rightHand["posScaled"]["x"] * self.webcam.frameSize[0])
            rightY = int(hands.rightHand["posScaled"]["y"] * self.webcam.frameSize[1])
            leftX = int(hands.leftHand["posScaled"]["x"] * self.webcam.frameSize[0])
            leftY = int(hands.leftHand["posScaled"]["y"] * self.webcam.frameSize[1])
            roiCenter = (int((rightX + leftX) / 2), int((rightY + leftY) / 2))
            decayFactor = 1 / abs(hands.rightHand["posScaled"]["x"] - hands.leftHand["posScaled"]["x"])

            # Get the image dimensions
            height, width = int(self.webcam.frameSize[1]), int(self.webcam.frameSize[0])
        
            # Create coordinate grid
            Y, X = np.indices((height, width))
            
            # Calculate the distance from the center
            distance = np.sqrt((X - roiCenter[0])**2 + (Y - roiCenter[1])**2)
            
            # Normalize the distance to the range [0, 1]
            max_distance = np.max(distance)
            distance_normalized = distance / max_distance
            
            # Create a mask with a decaying brightness based on the distance
            mask = np.exp(-decayFactor * distance_normalized)
            
            # Convert the mask to 3 channels (for colored images)
            mask = np.repeat(mask[:, :, np.newaxis], 3, axis=2)
            
            # Apply the mask to the image
            result = frame * mask
            
            # Convert result to uint8
            frame = np.clip(result, 0, 255).astype(np.uint8)
        
        return frame

    # blur the frame for reverb animation
    def blurFrame(self, frame, reverb):
        if reverb > 0.2:
            ksize = int(60 * reverb)
            ksize += (ksize + 1) % 2
            blur_kernel = np.zeros((ksize, ksize))
            for i in range(ksize):
                blur_kernel[i, i] = 1.0
            blur_kernel /= ksize
            frame = cv2.filter2D(frame, ddepth=-1, kernel=blur_kernel)
        return frame

    # draw volume bar based on the decibel value
    def drawVolumeBar(self, frame, db):
        volScaled = max(0, min(db / 80.0, 1))
        bottomPosY = int(self.webcam.frameSize[1] - 100)
        cv2.rectangle(frame, VOLUME_BAR_POS, (VOLUME_BAR_POS[0] + 100, bottomPosY), (169, 169, 169), -1)
        
        volLevel = max(int(bottomPosY + (VOLUME_BAR_POS[1] - bottomPosY) * volScaled), VOLUME_BAR_POS[1])
        cv2.rectangle(frame, (VOLUME_BAR_POS[0], volLevel), (VOLUME_BAR_POS[0] + 100, bottomPosY), self.getVolumeColor(volScaled), -1)
        return frame
    
    # change the color of the volume bar depending on the volume level
    def getVolumeColor(self, t):
        if t < 0.5:
            red = int(255 * (2 * t))
            green = 255
        else:
            red = 255
            green = int(255 * (2 * (1 - t)))
        return (0, green, red)
    
    # draw icons for currently detected hands
    def drawDetectedHands(self, frame):
        rightHand = self.gestureRecognition.rightHand
        leftHand = self.gestureRecognition.leftHand

        if rightHand["gesture"] != None:
            x = int(self.webcam.frameSize[0] - HAND_SIZE[0])
            y = 0

            for c in range(0, 3):
                frame[y:y + HAND_SIZE[1], x:x + HAND_SIZE[0], c] = (RIGHT_HAND_ALPHA / 255.0) * RIGHT_HAND[:, :, c] + (1 - RIGHT_HAND_ALPHA / 255.0) * frame[y:y + HAND_SIZE[1], x:x + HAND_SIZE[0], c]

        if leftHand["gesture"] != None:
            x = 0
            y = 0

            for c in range(0, 3):
                frame[y:y + HAND_SIZE[1], x:x + HAND_SIZE[0], c] = (LEFT_HAND_ALPHA / 255.0) * LEFT_HAND[:, :, c] + (1 - LEFT_HAND_ALPHA / 255.0) * frame[y:y + HAND_SIZE[1], x:x + HAND_SIZE[0], c]

        return frame

    # draw pointing right hand for cursor
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

    # draw hovering animation
    def drawCircularSector(self, frame, duration):
        if duration > 0:
            hand = self.gestureRecognition.rightHand
            x = int(hand["tipPosScaled"]["x"] * self.webcam.frameSize[0])
            y = int(hand["tipPosScaled"]["y"] * self.webcam.frameSize[1])
            endAngle = int(360 * duration / HOVER_THRESH)
            cv2.ellipse(frame, (x, y), (RADIUS, RADIUS), 0, 0, endAngle, (255, 255, 255), 15)
        
        return frame

    # checks if hover duration is above threshold for pausing
    def checkPauseHover(self):
        hand = self.gestureRecognition.rightHand
        if hand["gesture"] == "Pointing_Up":
            self.pauseHoverCount += 1
        else:
            self.pauseHoverCount = 0
        
        return self.pauseHoverCount > HOVER_THRESH
    
    # checks if hover duration is above threshold for quitting
    def checkQuitHover(self):
        hand = self.gestureRecognition.rightHand
        if hand["gesture"] == "Victory":
            self.quitHoverCount += 1
        else:
            self.quitHoverCount = 0
        
        return self.quitHoverCount > HOVER_THRESH

    # checks if hover duration is above threshold for music selection
    def checkMusicHover(self):
        hand = self.gestureRecognition.rightHand
        if hand["gesture"] == "Pointing_Up":
            tipPosFrame = (int(hand["tipPosScaled"]["x"] * self.webcam.frameSize[0]), int(hand["tipPosScaled"]["y"] * self.webcam.frameSize[1]))
            if tipPosFrame[0] > self.webcam.frameSize[0] - COVER_SIZE - GAP_SIZE:
                if tipPosFrame[1] < COVER_SIZE + GAP_SIZE:
                    if self.musicHoverCount["posId"] == 0:
                        self.musicHoverCount["count"] += 1
                    else:
                        self.musicHoverCount["posId"] = 0
                        self.musicHoverCount["count"] = 0
                elif COVER_SIZE + GAP_SIZE*2 < tipPosFrame[1] and tipPosFrame[1] < COVER_SIZE*2 + GAP_SIZE*2:
                    if self.musicHoverCount["posId"] == 1:
                        self.musicHoverCount["count"] += 1
                    else:
                        self.musicHoverCount["posId"] = 1
                        self.musicHoverCount["count"] = 0
                elif COVER_SIZE*2 + GAP_SIZE*3 < tipPosFrame[1] and tipPosFrame[1] < COVER_SIZE*3 + GAP_SIZE*4:
                    if self.musicHoverCount["posId"] == 2:
                        self.musicHoverCount["count"] += 1
                    else:
                        self.musicHoverCount["posId"] = 2
                        self.musicHoverCount["count"] = 0
            else:
                self.musicHoverCount["posId"] = None
                self.musicHoverCount["count"] = 0
        else:
            self.musicHoverCount["posId"] = None
            self.musicHoverCount["count"] = 0


        if self.musicHoverCount["count"] > HOVER_THRESH:
            musicFile = self.musicList[self.musicHoverCount["posId"]]
            selectedMusic = os.path.join(MUSIC_DIR, musicFile)
        else:
            selectedMusic = None
        
        return selectedMusic