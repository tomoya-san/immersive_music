from src2.music_player import MusicPlayer

import threading

MUSIC_PATH = "music/gen_alpha.mp3"

def main():
    musicPlayer = MusicPlayer()
    musicPlayer.setMusic(MUSIC_PATH)

    thread1 = threading.Thread(target=musicPlayer.play, daemon=False)
    thread1.start()

    while True:
        user_input = input("Enter something (or 'exit' to quit): ")
    
        musicPlayer.setReverbRoomSize(float(user_input))
        
        print(f"Room size: {user_input}")

if __name__ == "__main__":
    main()