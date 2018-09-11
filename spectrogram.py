
import wave
import pylab
import multiprocessing
import os 
from os.path import isfile, join
import gc

sounds_path = "./all/audio_test/"
data_path = "./data/audio_test/"

def graph_spectrogram(wav_file):
    sound_info, frame_rate = get_wav_info(sounds_path+wav_file+".wav")
    pylab.figure(figsize=(2, 2), dpi=100, frameon=False)
    pylab.specgram(sound_info, Fs=frame_rate)
    pylab.subplots_adjust(bottom = 0)
    pylab.subplots_adjust(top = 1)
    pylab.subplots_adjust(right = 1)
    pylab.subplots_adjust(left = 0)
    pylab.savefig(data_path+wav_file+".png", dpi=100)
    pylab.close("all")
    gc.collect
def get_wav_info(wav_file):
    wav = wave.open(wav_file, 'r')
    frames = wav.readframes(-1)
    sound_info = pylab.fromstring(frames, 'int16')
    frame_rate = wav.getframerate()
    wav.close()
    return sound_info, frame_rate

sounds_file = [os.path.splitext(f)[0] for f in os.listdir(sounds_path) if isfile(join(sounds_path, f))]
data_file = [os.path.splitext(f)[0] for f in os.listdir(data_path) if isfile(join(data_path, f))]

files = list(set(sounds_file) - set(data_file))

pool = multiprocessing.Pool(processes=2)
pool.map(graph_spectrogram,files)
