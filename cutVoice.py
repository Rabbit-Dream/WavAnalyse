import os
import math
import wave
import numpy as np
import pylab as pl
import threading
import re
# from multiprocessing import process


import math
import numpy as np
from scipy.io import wavfile

from vad import *

# 参数设置

frame_duration_ms = 30
padding_duration_ms = 300
frameSize = 256
overLap = 0

# def ZeroCR(waveData,frameSize,overLap):

# ============ test the algorithm =============
# read wave file and get parameters.
def CutSave(path):

    filename = path
    WAVE = wave.open(filename)
    nchannels, sampwidth, framerate, nframes = WAVE.getparams()[:4]
    WAVE.close()

    sample_time = 1/framerate           # 采样点的时间间隔
    time = nframes/framerate            # 声音信号的长度
    sample_frequency, audio_sequence = wavfile.read(filename)
    x_seq = np.arange(0,time,sample_time)

    wave_data=audio_sequence
    wave_data.shape = -1, 1

    wlen = len(wave_data)
    step = frameSize - overLap
    frameNum = math.ceil(wlen/step)

    dirname0="".join(filename.split("."))
    dirname = dirname0.replace('F','G')
    if not os.path.exists(dirname):
        os.mkdir(dirname)
    # print(dirname)
    # print(type(dirname))

    nowdata=np.zeros((1,1),dtype=np.int8)
    stepnum=0
    amps=[]
    czrs=[]
    try:
        for i in range(frameNum):

            curFrame = wave_data[np.arange(i*step,min(i*step+frameSize,wlen))]
            temp = curFrame - np.mean(curFrame)
            nowzcr = sum(temp[0:-1]*temp[1::]<=0)[0]
            czrs.append(nowzcr)

            if nowzcr<200:
                nowdata=np.append(nowdata,curFrame)
            else:
                if len(nowdata)>framerate*1: #时常大于1秒存储

                    vad = webrtcvad.Vad(0)
                    frames = frame_generator(30, nowdata, framerate)
                    frames = list(frames)
                    segments = vad_collector(framerate, frame_duration_ms, padding_duration_ms, vad, frames)
                    out=b''.join(segments)

                    with wave.open(dirname+"/"+str(stepnum)+".wav","w") as f:
                        f.setnchannels(nchannels)
                        f.setsampwidth(sampwidth)
                        f.setframerate(framerate)
                        f.writeframes(out)

                    stepnum+=1
                    nowdata=np.zeros((1,1),dtype=np.int8)
    except Exception as e:
        print(filename, e)

def main():
    Path = 'F:\\September\\'
    for dir in os.listdir(Path):
        t = threading.Thread(target=test,args=(Path+'\\'+dir,))
        t.start()


def test(filename):
    # tasks = []
    for file in os.listdir(filename):
        path = filename +'\\'+ file
        CutSave(path)
        # tasks.append(threading.Thread(target=CutSave,args=(path,)))

    # for index, task in enumerate(tasks):
    #     task.start()
    #     print(f'task {index} start')


if __name__ == '__main__':
    main()
            