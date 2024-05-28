import numpy as mp
import pyaudio as pa
import struct
import matplotlib.pyplot as plt

CHUNK = 1024 * 2
FORMAT = pa.paInt16
CHANNELS = 1
RATE = 44100 # Hz

p = pa.PyAudio()

stream = p.open(
    format=FORMAT,
    channels=CHANNELS,
    rate=RATE,
    input=True,
    output=True,
    frames_per_buffer=CHUNK
)

data = stream.read(CHUNK)

dataInt = struct.unpack(str(CHUNK) + 'h', data)

print(dataInt)