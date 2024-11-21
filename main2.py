from src2.music_player import MusicPlayer
from src2.gesture_recognition import GestureRecognition
from src2.webcam import Webcam
from src2.music_interactor import MusicInteractor
from src2.menu import Menu

import threading
import cv2
import numpy as np

MUSIC_PATH = "music/gen_alpha.mp3"

def main():

    webcam = Webcam()
    webcam.initialize()

    gestureRecognition = GestureRecognition()
    gestureRecognition.initialize()

    menu = Menu(webcam, gestureRecognition)

    musicPlayer = MusicPlayer()
    musicThread = threading.Thread(target=musicPlayer.play, daemon=True)

    musicInteractor = MusicInteractor()

    while True:
        frame = webcam.getFrame()
        result = gestureRecognition.recognize(frame)
        gestureRecognition.saveResult(result)

        if menu.screenId == "select_music":
            frame = menu.selectMusic(frame)
            frame = menu.drawCircularSector(frame, menu.quitHoverCount)
            cv2.imshow("webcam", frame)
            selectedMusic = menu.checkMusicHover()
            if selectedMusic != None:
                print(f"Selected: {selectedMusic}")
                musicPlayer.setMusic(selectedMusic)
                menu.screenId = "playing"
                musicThread.start()
              
        elif menu.screenId == "playing":
            frame = cv2.flip(frame, 1)
            frame = menu.blurFrame(frame, musicPlayer.reverb.room_size)
            frame = menu.darkenSurrounding(frame)
            frame = menu.drawVolumeBar(frame, musicPlayer.db)
            frame = menu.drawCircularSector(frame, menu.pauseHoverCount)
            frame = menu.drawCircularSector(frame, menu.quitHoverCount)
            cv2.imshow("webcam", frame)
            musicInteractor.hand2gain(gestureRecognition.rightHand)
            musicInteractor.hand2reverb(gestureRecognition.leftHand)
            musicInteractor.hand2filter(gestureRecognition.leftHand, gestureRecognition.rightHand)
            
            musicPlayer.setGain(musicInteractor.gain)
            musicPlayer.setReverbRoomSize(musicInteractor.roomSize)
            musicPlayer.setBandPassFilter(musicInteractor.low, musicInteractor.high)

            if menu.checkPauseHover():
                musicPlayer.paused = True
                menu.screenId = "pausing"
                menu.pauseHoverCount = 0

        elif menu.screenId == "pausing":
            frame = menu.pausingMusic(frame)
            frame = menu.drawCircularSector(frame, menu.pauseHoverCount)
            cv2.imshow("webcam", frame)

            if menu.checkPauseHover():
                musicPlayer.paused = False
                menu.screenId = "playing"
                menu.pauseHoverCount = 0

        if menu.checkQuitHover():
            print("Terminated.")
            break

        if cv2.waitKey(1) == ord('q'):
            print("Terminated")
            break
        

    
    webcam.terminate()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()