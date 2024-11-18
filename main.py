from src.gesture_recognition import GestureRecognition
from src.music_player import MusicPlayer

import time
import tkinter as tk

import threading
import cv2
import math

def main():
    gestureRecognition = GestureRecognition()
    musicPlayer = MusicPlayer()

    def musicLoop():
        musicPlayer.start_non_blocking_processing()
        while(musicPlayer.stream.is_active()):
            #time.sleep(0.1)
            if gestureRecognition.leftGesture == "Pointing_Up":
                musicPlayer.setCutoff(gestureRecognition.rightOpenness, gestureRecognition.rightHandPos[0])
            print(f"High: {musicPlayer.highcut}")
            print(f"Low: {musicPlayer.lowcut}")
            
        musicPlayer.terminate_processing()
    
    thread1 = threading.Thread(target=musicLoop, daemon=True)
    thread1.start()

    while gestureRecognition.cap.isOpened():
        ret, frame = gestureRecognition.cap.read()

        if ret == False:
            print('Error: Could not receive frame.')
            break
        
        annotatedImage = gestureRecognition.recognize(frame)
        cv2.imshow("webcam", annotatedImage)

        # Press 'q' to close webcam window.
        if cv2.waitKey(1) == ord('q'):
            print("Closing webcam.")
            break

    gestureRecognition.cap.release()
    cv2.destroyAllWindows()

    

    


if __name__ == '__main__':
    main()




    # def onScaleHigh(value):
    #     musicPlayer.highcut = int(value)
    
    # def onScaleLow(value):
    #     musicPlayer.lowcut = int(value)

    # root = tk.Tk()
    # root.title("Bandpass Filter")
    # root.geometry("500x300")

    # scaleHigh = tk.Scale(root, from_=50, to=10000, resolution=50, command=onScaleHigh)
    # scaleHigh.pack()

    # scaleLow = tk.Scale(root, from_=50, to=10000, resolution=50, command=onScaleLow)
    # scaleLow.pack()

    # def musicLoop():
    #     musicPlayer.start_non_blocking_processing()
    #     while(musicPlayer.stream.is_active()):
    #         time.sleep(0.1)
    #         print(f"Right: {gesture_recognition.RIGHT_POS}")
    #         print(f"Left: {gesture_recognition.LEFT_POS}")
    #     musicPlayer.terminate_processing()

    # thread1 = threading.Thread(target=musicLoop, daemon=True)
    # thread1.start()
    
    # gestureRecognition.startRecognition()

    # #root.mainloop()