import wave
import pyaudio
import av
import numpy as np
import librosa
import scipy
import scipy.io.wavfile as wav
from scipy.signal import fftconvolve

MUSIC_PATH = "music/gen_alpha.mp3"
IR_PATH = "ir/little_basement.wav"
CHUNK_SIZE = 1024

DELAY_MS = 500
DECAY_FACTOR = 0.9
SAMPLE_RATE = 44100

np_to_pa_format = {
    np.dtype('float32') : pyaudio.paFloat32,
    np.dtype('int32') : pyaudio.paInt32,
    np.dtype('int16') : pyaudio.paInt16,
    np.dtype('int8') : pyaudio.paInt8,
    np.dtype('uint8') : pyaudio.paUInt8
}

class MusicPlayer():
    def __init__(self):
        self.input_array, self.sample_rate = librosa.load(MUSIC_PATH, sr=44100, dtype=np.float32, offset=0)
        self.cycle_count = 0
        self.lowcut = 20
        self.highcut = 300
        self.filter_state = np.zeros(4)
        self.delaySamples = int(SAMPLE_RATE * DELAY_MS / 1000)
        self.history = np.zeros(self.delaySamples)
    
    def setCutoff(self, handOpenness, handPos):
        centerFreq = 10 * 2 ** (handPos * 10)
        halfBandWidth = 10 * 2 ** (handOpenness * 10)
        self.lowcut = max(10, centerFreq - halfBandWidth)
        self.highcut = min(10000, centerFreq + halfBandWidth)
    
    def bandPassFilter(self, signal):
        fs = 44100
        nyq= 0.5 * fs
        low = self.lowcut / nyq
        high = self.highcut / nyq

        order = 2

        b, a = scipy.signal.butter(order, [low, high], 'bandpass', analog=False)

        y, self.filter_state = scipy.signal.lfilter(b, a, signal, axis=0, zi=self.filter_state) # NB: filtfilt needs forward and backward information to filter. So it can't be used in realtime filtering where i have no info about future samples! lfilter is better for real time applications!
        return y
    
    def reverb(self, signal):
        y = np.zeros_like(signal)
        for i in range(len(signal)):
            delayedSample = self.history[i % self.delaySamples]
            y[i] = signal[i] + delayedSample * DECAY_FACTOR
            self.history[i % self.delaySamples] = signal[i]
        return y
    
    def pyaudio_callback(self, in_data, frame_count, time_info, status):
        audio_size = np.shape(self.input_array)[0]

        if frame_count*self.cycle_count > audio_size:
            # Processing is complete.
            return (None, pyaudio.paComplete)
        
        elif frame_count*(self.cycle_count+1) > audio_size:
            # Last frame to process.
            frames_left = audio_size - frame_count*self.cycle_count

        else:
            # Every other frame.
            frames_left = frame_count

        data = self.input_array[frame_count*self.cycle_count:frame_count*self.cycle_count+frames_left]
        #data = self.bandPassFilter(data)
        #data = self.reverb(data)

        out_data = data.astype(np.float32).tobytes()
        self.cycle_count+=1
        return (out_data, pyaudio.paContinue)
    
    def start_non_blocking_processing(self, save_output=True, frame_count=2**10, listen_output=True):
        self.save_output = save_output
        self.frame_count = frame_count

        # Initiate PyAudio
        self.pa = pyaudio.PyAudio()
        # Open stream using callback
        self.stream = self.pa.open(format=np_to_pa_format[self.input_array.dtype],
                        channels=1,
                        rate=self.sample_rate,
                        output=listen_output,
                        input=not listen_output,
                        stream_callback=self.pyaudio_callback,
                        frames_per_buffer=frame_count)

        # Start the stream
        self.stream.start_stream()

    def terminate_processing(self):
        '''
        Terminates stream opened by self.start_non_blocking_processing.
        MUST be called AFTER self.processing returns False.
        '''
        # Stop stream.
        self.stream.stop_stream()
        self.stream.close()

        # Close PyAudio.
        self.pa.terminate()

        # Resets count.
        self.cycle_count = 0
        # Resets output.
        self.output_array = np.array([[], []], dtype=self.input_array.dtype).T

    def play(self):
        container = av.open(MUSIC_PATH)

        audio_stream = container.streams.audio[0]

        samplerate = audio_stream.rate 
        channels = audio_stream.channels
        print(f"Sample rate: {samplerate}")
        p = pyaudio.PyAudio()

        audio_device = p.open(format=pyaudio.paFloat32,
                            channels=channels,
                            rate=samplerate,
                            output=True)
        while True:
            try:
                frame = next(container.decode(audio=0))
                audio_data = frame.to_ndarray().astype('float32')
                interleaved_data = audio_data.T.flatten().tobytes()
                audio_device.write(interleaved_data)
                time = round(float(frame.pts * audio_stream.time_base), 2)
                #print(time) # display current time
            except (StopIteration, av.error.EOFError):
                break
            
        audio_stream.close()
        container.close()
        audio_device.stop_stream()
        audio_device.close()
        p.terminate()