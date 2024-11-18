import numpy as np
import soundfile as sf
import pyaudio
from pedalboard import Pedalboard, Reverb, Gain, Chorus, Compressor, Delay, PitchShift

# Path to the audio file
AUDIO_PATH = "music/gen_alpha.mp3"  # Replace with your file's path

# Initialize the pedalboard with a reverb effect
pedalboard = Pedalboard([
    #Compressor(threshold_db=-50, ratio=25),
    #Gain(gain_db=30),
    #Chorus(),
    #Phaser(),
    #Convolution("./guitar_amp.wav", 1.0),
    #Reverb(room_size=0.9),
])

# Parameters for streaming
sample_rate = 44100  # The sample rate of the audio
chunk_size = 128  # Number of frames per chunk (to simulate streaming)

# Open the audio file for reading (audio file must be in WAV, FLAC, or another compatible format)
audio_file = sf.SoundFile(AUDIO_PATH)
sample_rate = audio_file.samplerate
num_channels = audio_file.channels
print(f"sample rate: {sample_rate}")
print(f"channels: {num_channels}")

# Set up PyAudio for output
p = pyaudio.PyAudio()
stream_out = p.open(format=pyaudio.paFloat32,  # Use 32-bit float format for audio processing
                    channels=num_channels,  # Mono audio
                    rate=sample_rate,
                    output=True,
                    frames_per_buffer=chunk_size)

# Simulate streaming by processing the file in chunks
print("Starting audio streaming...")

try:
    while True:
        # Read a chunk from the audio file
        audio_chunk = audio_file.read(chunk_size)

        # If the chunk is empty, we've reached the end of the file
        if len(audio_chunk) == 0:
            break

        # Process the chunk with pedalboard (apply the reverb effect)
        processed_chunk = pedalboard(audio_chunk, sample_rate=sample_rate)

        # Output the processed chunk to speakers
        stream_out.write(processed_chunk.astype(np.float32).tobytes())

except KeyboardInterrupt:
    print("\nStopping audio streaming.")
finally:
    # Close the audio stream
    stream_out.stop_stream()
    stream_out.close()
    p.terminate()
    audio_file.close()
    print("Streaming stopped.")
