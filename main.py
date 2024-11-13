from src.GestureRecognition import GestureRecognition
from src.MusicPlayer import MusicPlayer

import time
import tkinter as tk

import threading


def main():
    #gestureRecognition = GestureRecognition()
    #gestureRecognition.startRecognition()
    musicPlayer = MusicPlayer()

    def onScaleHigh(value):
        musicPlayer.highcut = int(value)

    root = tk.Tk()
    root.title("Bandpass Filter")
    root.geometry("500x300")

    scaleHigh = tk.Scale(root, from_=100, to=1000, resolution=50, command=onScaleHigh)
    scaleHigh.pack()

    def musicLoop():
        musicPlayer.start_non_blocking_processing()
        while(musicPlayer.stream.is_active()):
            time.sleep(0.1)
        musicPlayer.terminate_processing()

    thread = threading.Thread(target=musicLoop, daemon=True)
    thread.start()

    root.mainloop()


if __name__ == '__main__':
    main()