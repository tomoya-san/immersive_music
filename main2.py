from src2.music_player import MusicPlayer
from src2.gesture_recognition import GestureRecognition
from src2.webcam import Webcam
from src2.music_interactor import MusicInteractor

import threading
import cv2

MUSIC_PATH = "music/gen_alpha.mp3"

def main():
    musicPlayer = MusicPlayer()
    musicPlayer.setMusic(MUSIC_PATH)

    thread1 = threading.Thread(target=musicPlayer.play, daemon=True)
    thread1.start()

    webcam = Webcam()
    webcam.initialize()

    gestureRecognition = GestureRecognition()
    gestureRecognition.initialize()

    musicInteractor = MusicInteractor()

    while True:
        frame = webcam.getFrame()
        result = gestureRecognition.recognize(frame)
        gestureRecognition.saveResult(result)
        cv2.imshow("webcam", cv2.flip(frame, 1))

        musicInteractor.hand2gain(gestureRecognition.rightHand)
        musicInteractor.hand2reverb(gestureRecognition.leftHand)
        musicInteractor.hand2filter(gestureRecognition.leftHand, gestureRecognition.rightHand)
        
        musicPlayer.setGain(musicInteractor.gain)
        musicPlayer.setReverbRoomSize(musicInteractor.roomSize)
        musicPlayer.setBandPassFilter(musicInteractor.low, musicInteractor.high)

        print(f"low: {musicInteractor.low}, high: {musicInteractor.high}")

        if cv2.waitKey(1) == ord('q'):
            print("Closing webcam.")
            break

    webcam.terminate()
    cv2.destroyAllWindows()

    # while True:
    #     user_input = input("Enter something (or 'exit' to quit): ")
    
    #     # musicPlayer.setReverbRoomSize(float(user_input))
    #     # print(f"Room size: {user_input}")

    #     # musicPlayer.setBandPassFilter(50, float(user_input))
    #     # print(f"High cut: {user_input}")

    #     musicPlayer.setGain(float(user_input))
    #     print(f"Gain: {user_input}")

if __name__ == "__main__":
    main()