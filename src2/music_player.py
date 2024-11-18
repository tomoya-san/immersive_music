from pedalboard import Pedalboard, Reverb
from pedalboard.io import AudioStream, AudioFile

BUFFER_SIZE = 12000

class MusicPlayer():
    def __init__(self):
        self.music = None
        self.reverb = Reverb(room_size=0.0)

        self.pedalboard = Pedalboard([
            self.reverb,
        ])
    
    # set music path
    def setMusic(self, music):
        self.music = music
    
    # play music in stream
    def play(self):
        with AudioStream(output_device_name=AudioStream.default_output_device_name) as stream:
            with AudioFile(self.music) as file:
                while file.tell() < file.frames:
                    audio_chunk = file.read(BUFFER_SIZE)
                    audio_chunk = self.pedalboard.process(audio_chunk, file.samplerate, reset=False)
                    stream.write(audio_chunk, file.samplerate)
    
    # set reverb room_size from 0.0 to 1.0
    def setReverbRoomSize(self, roomSize):
        self.reverb.room_size = roomSize
    