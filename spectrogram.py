
import wave
import pylab
import multiprocessing
import os 
from os.path import isfile, join
import gc

sounds_path = "./all/"
data_path = "./data/"

def graph_spectrogram(wav_file):
    sound_info, frame_rate = get_wav_info(sounds_path+wav_file+".wav")
    pylab.figure(num=None, figsize=(20, 20))
    pylab.specgram(sound_info, Fs=frame_rate)
    pylab.axis("off")
    pylab.savefig(data_path+wav_file+".png", bbox_inches='tight', dpi=300, pad_inches=0)
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

pool = multiprocessing.Pool()
pool.map(graph_spectrogram,files)