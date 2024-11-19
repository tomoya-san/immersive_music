import cv2

class Webcam():
    def __init__(self):
        self.cap = None
        self.frameSize = None
        self.frameRate = None
    
    def initialize(self):
        self.cap = cv2.VideoCapture(1)

        if (self.cap.isOpened()):
            print("Successfully opened webcam.")
        else:
            print("Error: Could not open webcam.")
            exit(0)

        self.frameSize = (self.cap.get(cv2.CAP_PROP_FRAME_WIDTH), self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.frameRate = int(self.cap.get(cv2.CAP_PROP_FPS))
        print(f"Frame Size: {self.frameSize}")
        print(f"Frame Rate: {self.frameRate}")
    
    def getFrame(self):
        if self.cap.isOpened():
            ret, frame = self.cap.read()
        else:
            print("Error: Webcam is not open anymore.")
            exit(0)
        
        if ret == False:
            print("Error: Could not receive frame.")
            exit(0)
        
        return frame

    def terminate(self):
        self.cap.release()