from pedalboard import Pedalboard, Reverb, LowpassFilter, HighpassFilter, Gain
from pedalboard.io import AudioStream, AudioFile, ReadableAudioFile

import numpy as np
import time

BUFFER_SIZE = 1200

class MusicPlayer():
    def __init__(self):
        self.music = None
        self.reverb = Reverb(room_size=0.0)
        self.lowpassFilter = LowpassFilter(cutoff_frequency_hz=10000.0)
        self.highpassFilter = HighpassFilter(cutoff_frequency_hz=50.0)
        self.gain = Gain(gain_db=1.0)

        self.pedalboard = Pedalboard([
            self.reverb,
            self.lowpassFilter,
            self.highpassFilter,
            self.gain,
        ])

        self.rms = 0.001
        self.currentFrame = 0
        self.paused = True
    
    # set music path
    def setMusic(self, music):
        self.music = music
    
    # play music in stream
    def play(self):
        self.paused = False
        self.currentFrame = 0
        with AudioStream(output_device_name=AudioStream.default_output_device_name) as stream:
            print(f"Sound device operating at SR={stream.sample_rate}")
            with ReadableAudioFile(self.music) as file:
                file = file.resampled_to(stream.sample_rate)
                file.seek(self.currentFrame)

                while file.tell() < file.frames:
                    if self.paused:
                        time.sleep(0.01)
                        continue

                    audio_chunk = file.read(BUFFER_SIZE)
                    audio_chunk = self.pedalboard.process(audio_chunk, file.samplerate, reset=False)
                    self.getRMS(audio_chunk)
                    self.getDecibel(0.001)
                    stream.write(audio_chunk, file.samplerate)
                    self.currentFrame = file.tell()
    
    # get loudness of audio chunk
    def getRMS(self, audioChunk):
        rms = np.sqrt(np.mean(audioChunk ** 2))
        self.rms = rms
    
    def getDecibel(self, refVal):
        db = 20 * np.log10(self.rms / refVal)
        self.db = db

    # set reverb room_size from 0.0 to 1.0
    def setReverbRoomSize(self, roomSize):
        self.reverb.room_size = roomSize
    
    # set band pass filter band width
    def setBandPassFilter(self, low, high):
        self.lowpassFilter.cutoff_frequency_hz = high
        self.highpassFilter.cutoff_frequency_hz = low

    # set gain
    def setGain(self, gain):
        self.gain.gain_db = gain