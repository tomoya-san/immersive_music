from pedalboard import Pedalboard, Reverb, Gain, Chorus, Compressor, Delay, PitchShift, Distortion
from pedalboard.io import AudioStream, AudioFile
import numpy as np

# Set initial gain value
gain_value = 1.0  # Starting gain (1.0 means no change)

board = Pedalboard([
    #Delay(delay_seconds=0.1, feedback=0.0, mix = 0.5)
    #PitchShift(1.0)
    Reverb()
])
#gain = board[0]
#reverb = board[1]

# Play an audio file by looping through it in chunks:
with AudioStream(output_device_name=AudioStream.default_output_device_name) as stream:
    with AudioFile("music/gen_alpha.mp3") as f:
        while f.tell() < f.frames:
            # Optionally, change the gain value dynamically during playback
            # For example, increase gain slowly over time
            gain_value = 1.0 + (f.tell() / f.frames) * 500  # Gradually increase gain

            #print(gain_value)
            # Add a Gain plugin to the stream
            #gain.gain_db = gain_value

            #reverb.room_size = 1.0
            #reverb.wet_level = 1.0

            print("repeat")
            # Decode and play 512 samples at a time
            audio_chunk = f.read(48000)

            audio_chunk = board.process(audio_chunk, 48000.00, reset=False)
            stream.write(audio_chunk, 48000.00)